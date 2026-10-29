"""Small CLI runtime: model artifacts are separate from evaluation runs."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from collections.abc import Callable
import yaml

from audio_df.data.manifests import load_manifest, uniform_stratified_subset
from audio_df.helpers.artifacts import artifacts_dir, platt, save_threshold
from audio_df.helpers.cache import feature_cache_dir
from audio_df.evaluation import evaluate_scores
from audio_df.helpers.eval_audio import trimmed_manifest, TRIM_RECIPE




def _timestamp() -> str:
    # UTC timestamp, microseconds avoid collisions in normal use.
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _new_directory(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        suffix = f"-{attempt}" if attempt else ""
        candidate = parent / f"{prefix}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Unable to create a unique run directory")


def _dataset_key(dataset: str) -> str:
    if dataset.lower() in {"2019la", "2019", "asvspoof2019"}:
        return "2019la"
    return dataset.lower()


class Run:
    def __init__(
        self, *, name: str, dataset: str = "2019LA", output: str = "runs",
        root: str = ".", seed: int = 1337, testsets: str = "eval",
        threshold_from: str = "dev",
        model_from: str | None = None, command: str = "run",
    ) -> None:
        if dataset.lower() not in {
            "2019la", "2019", "asvspoof2019", "spoofceleb", "phonemedf",
        }:
            raise ValueError("Supported training datasets are 2019LA, SpoofCeleb and PhonemeDF")
        if command not in {"train", "evaluate", "run"}:
            raise ValueError(f"Unknown command: {command}")
        self.name = name
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.output_root = (self.root / output).resolve()
        self.artifact_root = artifacts_dir(self.root) / name
        self.seed = seed
        self.testsets = [part.strip() for part in testsets.split(",") if part.strip()]
        self.threshold_from = threshold_from.strip()
        if not self.threshold_from or not self.threshold_from.replace("_", "").isalnum():
            raise ValueError("Invalid --threshold-from name")
        if not self.testsets or len(self.testsets) != len(set(self.testsets)):
            raise ValueError("--testsets must contain unique test-set names")
        self.command = command
        self._model_from_arg = model_from
        self._artifact_dir: Path | None = None
        self._run_dir: Path | None = None
        self._train_preprocessing: str | None = None
        self._train_subset_info: dict[str, Any] | None = None
        self._subset_dataset = (
            self.dataset.lower()
            if self.dataset.lower() in {"spoofceleb", "phonemedf"}
            else None
        )
        self._training_subset_per_stratum, self._training_subset_strata = (
            self._load_training_subset_config(self._subset_dataset)
            if self._subset_dataset
            else (100, ["label", "attack_id"])
        )

    def _load_training_subset_config(self, dataset: str) -> tuple[int, list[str]]:
        default = (100, ["label", "attack_id"])
        config_path = self.root / "configs" / "data.yaml"
        if not config_path.is_file():
            return default
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        subset = config.get(dataset, {}).get("training_subset", {})
        per_stratum = subset.get("max_per_stratum", default[0])
        strata = subset.get("strata", default[1])
        if not isinstance(per_stratum, int) or per_stratum < 1:
            raise ValueError(
                f"{dataset}.training_subset.max_per_stratum must be >= 1"
            )
        if not isinstance(strata, list) or not strata or any(not isinstance(col, str) for col in strata):
            raise ValueError(
                f"{dataset}.training_subset.strata must be a non-empty list of columns"
            )
        return per_stratum, strata

    @classmethod
    def cli(cls, dataset: str = "2019LA") -> Callable[..., Any]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            # Derive from the decorated function's defining file, not call stack.
            name = Path(inspect.getfile(func)).stem
            signature = inspect.signature(func)
            # Resolve --root even for --help so available manifests reflect the chosen project.
            root_parser = argparse.ArgumentParser(add_help=False)
            root_parser.add_argument("--root", default=".")
            

            @wraps(func)
            def invoke(argv: list[str] | None = None) -> Any:
                
                help_root, _ = root_parser.parse_known_args(argv)
                manifest_dir = Path(help_root.root) / "data" / "manifests"
                available = sorted(p.stem for p in manifest_dir.glob("*.csv") if p.stem != "train")
                calibrations = ", ".join(available) or "none (run data-prepare)"
                tests = ", ".join(available) or "none (run data-prepare)"
                parser = argparse.ArgumentParser(description=func.__doc__)
                parser.add_argument(
                    "command",
                    nargs="?",
                    choices=("run", "train", "evaluate"),
                    default="run",
                    help="run: train + evaluate; train: save model; evaluate: load model + score",
                )
                parser.add_argument(
                    "--i",
                    default=dataset,
                    metavar="DATASET",
                    help=(
                        f"Training dataset identifier (default: {dataset}). "
                        "Available: 2019LA, SpoofCeleb, PhonemeDF"
                    ),
                )
                parser.add_argument(
                    "--o",
                    default="runs",
                    metavar="DIR",
                    help="Evaluation output directory (default: runs/)",
                )
                parser.add_argument(
                    "--root",
                    default=".",
                    metavar="DIR",
                    help="Project root containing data/, configs/ and artifacts/ (default: current directory)",
                )
                parser.add_argument(
                    "--seed",
                    type=int,
                    default=1337,
                    metavar="N",
                    help="Random seed for training (default: 1337)",
                )
                parser.add_argument(
                    "--testsets",
                    default="eval",
                    metavar="NAMES",
                    help=(
                            "Comma-separated final test sets (default: eval). "
                        f"Available manifests: {tests}"
                    )
                )
                parser.add_argument(
                    "--threshold-from",
                    default="dev",
                    metavar="NAME",
                    help=f"Validation manifest for threshold calibration (default: dev). Available: {calibrations}",
                )
                parser.add_argument(
                    "--model-from",
                    default=None,
                    metavar="PATH",
                    help="Trained model artifact directory (default: latest completed model)",
                )               
                for param_name, param in signature.parameters.items():
                    if param_name == "run" or param.default is inspect.Parameter.empty:
                        continue
                    if isinstance(param.default, bool):
                        parser.add_argument(f"--{param_name.replace('_', '-')}", action="store_false" if param.default else "store_true")
                    else:
                        parser.add_argument(f"--{param_name.replace('_', '-')}", type=type(param.default), default=param.default)
                args = parser.parse_args(argv)
                run = cls(name=name, dataset=args.i, output=args.o, root=args.root,
                          seed=args.seed, testsets=args.testsets,
                          threshold_from=args.threshold_from,
                          model_from=args.model_from, command=args.command)
                kwargs = {key: getattr(args, key) for key in signature.parameters if key != "run"}
                return func(run, **kwargs)

            return invoke
        return decorate

    @property
    def model_from(self) -> Path:
        if self._artifact_dir is not None:
            return self._artifact_dir
        if self._model_from_arg:
            chosen = Path(self._model_from_arg)
            chosen = (self.root / chosen).resolve()
            if not (chosen / "COMPLETE").exists():
                raise FileNotFoundError(f"Model artifact is not marked complete: {chosen}")
            self._validate_artifact_dataset(chosen)
            return chosen
        if not self.artifact_root.exists():
            raise FileNotFoundError(f"No {self.name} model found. Train first.")
        valid = sorted(
            p
            for p in self.artifact_root.iterdir()
            if p.is_dir()
            and (p / "COMPLETE").is_file()
            and self._artifact_dataset(p) == _dataset_key(self.dataset)
        )
        if not valid:
            raise FileNotFoundError(
                f"No completed {self.name} model for dataset {self.dataset!r} found. Train first."
            )
        return valid[-1]

    def _artifact_dataset(self, artifact: Path) -> str:
        metadata_path = artifact / "preprocessing.json"
        if not metadata_path.is_file():
            return "2019la"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return _dataset_key(metadata.get("dataset", "2019LA"))

    def _validate_artifact_dataset(self, artifact: Path) -> None:
        artifact_dataset = self._artifact_dataset(artifact)
        if artifact_dataset != _dataset_key(self.dataset):
            raise ValueError(
                f"Model artifact {artifact} was trained on dataset {artifact_dataset!r}, "
                f"not {self.dataset!r}"
            )

    def model_from_for(self, architecture: str) -> Path:
        """Resolve a completed artifact of another architecture, respecting --model-from."""
        if not architecture or not architecture.replace("_", "").isalnum():
            raise ValueError("Invalid architecture")
        if self._model_from_arg:
            selected = (self.root / self._model_from_arg).resolve()
            if not (selected / "COMPLETE").is_file():
                raise FileNotFoundError(f"Model artifact is not complete: {selected}")
            if selected.parent != (artifacts_dir(self.root) / architecture).resolve():
                raise ValueError(f"--model-from must select an {architecture} artifact")
            self._validate_artifact_dataset(selected)
            return selected
        parent = artifacts_dir(self.root) / architecture
        valid = (
            sorted(
                p for p in parent.iterdir()
                if p.is_dir()
                and (p / "COMPLETE").is_file()
                and self._artifact_dataset(p) == _dataset_key(self.dataset)
            )
            if parent.is_dir()
            else []
        )
        if not valid:
            raise FileNotFoundError(
                f"No completed {architecture} artifact for dataset {self.dataset!r} found"
            )
        return valid[-1]

    @property
    def output(self) -> Path:
        if self._run_dir is None:
            self._run_dir = _new_directory(self.output_root, f"E-{_timestamp()}-{self.name}")
        return self._run_dir

    def manifest(self, split: str):
        if self._subset_dataset:
            split_files = {
                "train": f"{self._subset_dataset}_train.csv",
                "dev": f"{self._subset_dataset}_development.csv",
                "development": f"{self._subset_dataset}_development.csv",
                "eval": f"{self._subset_dataset}_evaluation.csv",
                "evaluation": f"{self._subset_dataset}_evaluation.csv",
            }
            filename = split_files.get(split, f"{self._subset_dataset}_{split}.csv")
        else:
            filename = f"{split}.csv"
        return load_manifest(self.root / "data" / "manifests" / filename, self.root)

    def train(self, model: Any) -> Any:
        # Match evaluation preprocessing: raw 2019LA silence duration leaks the label.
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace)
        params = inspect.signature(model.fit).parameters
        extra: dict[str, Any] = {}
        if self._subset_dataset:
            train_df = self._dataset_training_manifest("train")
            if "dev" in params:
                extra["dev"] = self._dataset_training_manifest("dev")
            self._train_subset_info = {
                "max_per_stratum": self._training_subset_per_stratum,
                "strata": self._training_subset_strata,
                "seed": self.seed,
            }
        else:
            train_df = self.manifest("train")
            if "dev" in params:
                extra["dev"] = self.manifest("dev")
        trimmed = self._is_trimmed("dev")
        if trimmed:
            train_df = trimmed_manifest(train_df, self.root / "data" / "cache", "train")
            if "dev" in extra:
                extra["dev"] = trimmed_manifest(extra["dev"], self.root / "data" / "cache", "dev")
        self._train_preprocessing = "trimmed" if trimmed else "raw"
        print(f"\n[{self.name.upper()}] TRAIN | dataset={self.dataset} | seed={self.seed}", flush=True)
        print(f"  train: {len(train_df):,} clips | {self._train_preprocessing}", flush=True)
        if self._train_subset_info:
            print(
                f"  uniform subset: max {self._training_subset_per_stratum} per "
                f"{self._training_subset_strata} stratum | seed={self.seed}",
                flush=True,
            )
            for split, frame in (("train", train_df), *extra.items()):
                counts = frame.groupby(
                    self._training_subset_strata, dropna=False
                ).size()
                distribution = ", ".join(
                    f"{key if isinstance(key, tuple) else (key,)}={count}"
                    for key, count in counts.items()
                )
                print(f"  {split} strata: {distribution}", flush=True)
        if "dev" in extra:
            print(f"  dev:   {len(extra['dev']):,} clips | {self._train_preprocessing} (training validation)", flush=True)
        print(f"  feature cache: {cache_dir}", flush=True)
        print(f"  artifact destination: {self.artifact_root}", flush=True)
        return model.fit(train_df, cache_dir, seed=self.seed, **extra)

    def _dataset_training_manifest(self, split: str):
        dataset = self._subset_dataset
        if dataset is None:
            raise RuntimeError("Training subsets are only available for configured datasets")
        source_filename = {
            "train": f"{dataset}_train.csv",
            "dev": f"{dataset}_development.csv",
        }[split]
        source_path = self.root / "data" / "manifests" / source_filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing {dataset} {split} manifest: {source_path}. "
                f"Run `uv run data-prepare --dataset {dataset}` first."
            )

        per_stratum = self._training_subset_per_stratum
        subset_dir = self.root / "data" / "cache" / dataset / "training_subsets"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / f"{split}_per-stratum-{per_stratum}_seed-{self.seed}.csv"
        metadata_path = subset_path.with_suffix(".json")
        stat = source_path.stat()
        expected_metadata = {
            "source": source_filename,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "max_per_stratum": per_stratum,
            "strata": self._training_subset_strata,
            "seed": self.seed,
        }
        if subset_path.is_file() and metadata_path.is_file():
            try:
                if json.loads(metadata_path.read_text(encoding="utf-8")) == expected_metadata:
                    return load_manifest(subset_path, self.root)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        import pandas as pd

        source = pd.read_csv(source_path)
        subset = uniform_stratified_subset(
            source,
            max_per_stratum=per_stratum,
            strata=self._training_subset_strata,
            seed=self.seed,
        )
        temporary_path = subset_path.with_suffix(".csv.tmp")
        subset.to_csv(temporary_path, index=False)
        os.replace(temporary_path, subset_path)
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(expected_metadata, indent=2), encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata_path)
        return load_manifest(subset_path, self.root)

    def _spoofceleb_training_manifest(self, split: str):
        """Backward-compatible helper alias for existing SpoofCeleb callers/tests."""
        return self._dataset_training_manifest(split)

    def save(self, model: Any) -> Path:
        directory = _new_directory(self.artifact_root, _timestamp())
        model.save(directory)
        preprocessing: dict[str, Any] = {
            "train": self._train_preprocessing,
            "trim_recipe": TRIM_RECIPE if self._train_preprocessing == "trimmed" else None,
            "dataset": self.dataset,
        }
        if self._train_subset_info is not None:
            preprocessing["training_subset"] = self._train_subset_info
        (directory / "preprocessing.json").write_text(
            json.dumps(preprocessing, indent=2), encoding="utf-8"
        )
        # A completion marker is written LAST. Incomplete artifacts are ignored.
        (directory / "COMPLETE").write_text("ok\n", encoding="utf-8")
        self._artifact_dir = directory
        print(f"Model saved: {directory}")
        return directory
    
    def _project_path(self, path: Path) -> str:
        """Path relative to the project root, POSIX-style, so run records work on any machine."""
        path = Path(path).resolve()
        return path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)

    def validate_testsets(self) -> None:
        """Check registered manifests and reject calibration/test data leakage.

        Utterance IDs are dataset-specific, so source audio paths are checked too.
        Cross-dataset duplicates with different IDs/paths require a separate audit.
        """
        manifest_dir = self.root / "data" / "manifests"
        available = {p.stem for p in manifest_dir.glob("*.csv")}
        if self._subset_dataset:
            aliases = {"development": "dev", "evaluation": "eval"}
            available = {
                alias
                for alias, filename in (
                    ("dev", f"{self._subset_dataset}_development.csv"),
                    ("eval", f"{self._subset_dataset}_evaluation.csv"),
                )
                if (manifest_dir / filename).is_file()
            }
            requested = {
                aliases.get(name, name) for name in (*self.testsets, self.threshold_from)
            }
        else:
            requested = set(self.testsets) | {self.threshold_from}
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Unknown manifests: {sorted(unknown)}. Available: {sorted(available)}"
            )
        threshold_from = (
            {"development": "dev", "evaluation": "eval"}.get(
                self.threshold_from, self.threshold_from
            )
            if self._subset_dataset
            else self.threshold_from
        )
        testsets = (
            [
                {"development": "dev", "evaluation": "eval"}.get(name, name)
                for name in self.testsets
            ]
            if self._subset_dataset
            else self.testsets
        )
        if len(testsets) != len(set(testsets)):
            raise ValueError("--testsets must contain unique dataset splits")
        if threshold_from in testsets:
            raise ValueError(
                f"Calibration set {self.threshold_from!r} cannot appear in --testsets"
            )
        # eval_clean10k and degraded variants derive from 2019 LA eval.
        # Prevent known overlap even when audio paths differ after degradation.
        def eval_family(name: str) -> bool:
            return name == "eval" or name.startswith("eval_")
        if (
            self.dataset.lower() in {"2019la", "2019", "asvspoof2019"}
            and eval_family(self.threshold_from)
            and any(eval_family(n) for n in self.testsets)
        ):
            raise ValueError(
                "Calibration and test sets from the ASVspoof 2019 LA eval family "
                "may overlap. Use dev for calibration."
            )
        calibration = self.manifest(self.threshold_from)
        cal_ids = set(calibration["utt_id"].astype(str))
        cal_paths = set(calibration["path"].astype(str))
        for name in self.testsets:
            test = self.manifest(name)
            if cal_ids.intersection(test["utt_id"].astype(str)) or cal_paths.intersection(test["path"].astype(str)):
                raise ValueError(f"Calibration set {self.threshold_from!r} overlaps test set {name!r}")

    def _is_trimmed(self, name: str) -> bool:
        return self.dataset.lower() in {"2019la", "2019", "asvspoof2019"} and (
            name == "dev" or name == "eval" or name.startswith("eval_")
        )

    def evaluate(self, model: Any) -> dict[str, Any]:
        self.validate_testsets()
        artifact = self.model_from
        output = self.output
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace)
        names = [self.threshold_from, *self.testsets]
        prepared = [(name, self.manifest(name), self._is_trimmed(name)) for name in names]
        print(f"\n[{self.name.upper()}] EVALUATE | dataset={self.dataset}", flush=True)
        print(f"  model artifact: {artifact}", flush=True)
        for name, df, trimmed in prepared:
            role = "threshold calibration" if name == self.threshold_from else "test"
            print(f"  {name}: {len(df):,} clips | {'trimmed' if trimmed else 'raw'} ({role})", flush=True)
        print(f"  feature cache: {cache_dir}", flush=True)
        print(f"  output: {output}", flush=True)
        scored = {}
        for name, df, trimmed in prepared:
            print(f"[{self.name.upper()}] Scoring {name} ({'trimmed' if trimmed else 'raw'})", flush=True)
            if trimmed:
                df = trimmed_manifest(df, self.root / "data" / "cache", name)
            scored[name] = model.score_manifest(
                df, output / "scores" / f"{name}.csv", split=name, cache_dir=cache_dir,
            )
        calibration = scored[self.threshold_from]
        results = {}
        for name in self.testsets:
            results[name] = evaluate_scores(calibration, scored[name], output / f"metrics_{name}.json")
            print(f"{name}: EER={results[name]['eer']:.4%}")
        (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        from audio_df.helpers.metrics import compute_eer
        cal_eer, threshold = compute_eer(calibration.score, calibration.label, higher_is_spoof=False)
        calibration_record = {
            "source": self.threshold_from, "eer": float(cal_eer),
            "eer_threshold": float(threshold), "n": len(calibration),
            "preprocessing": "trimmed" if self._is_trimmed(self.threshold_from) else "raw",
        }
        (output / "calibration.json").write_text(json.dumps(calibration_record, indent=2), encoding="utf-8")
        la_2021 = results.get("asvspoof2021_eval")  # error rate shown to arbitration judges
        if _dataset_key(self.dataset) == "2019la":  # the app serves 2019 LA-trained detectors only
            error_rate = round(100 * la_2021["eer"], 1) if la_2021 else None
            save_threshold(artifact, {**calibration_record, "platt": platt(calibration),
                                      "error_rate_pct": error_rate}, output.name)
        train_prep = Path(artifact) / "preprocessing.json"
        record = {
            "architecture": self.name, "dataset": self.dataset,
            "model_artifact": self._project_path(artifact), "testsets": self.testsets,
            "seed": self.seed,
            # Artifacts saved before preprocessing.json existed were trained on raw audio.
            "train_preprocessing": (json.loads(train_prep.read_text(encoding="utf-8"))["train"]
                                    if train_prep.exists() else "raw"),
            # D2/D3 only: which WavLM hidden state the attention used, so rounds stay distinguishable.
            **({"wavlm_layer": layer} if (layer := getattr(getattr(getattr(model, "head", None), "config", None),
                                                           "layer", None)) is not None else {}),
            "split_roles": {name: ("validation_threshold" if name == self.threshold_from else "held_out_test") for name in names},
            "reported_metrics": "held_out_test_only", "threshold_source": self.threshold_from,
            "evaluation_preprocessing": {
                "testsets": {name: ("trimmed" if trimmed else "raw") for name, _, trimmed in prepared},
                "trim_recipe": TRIM_RECIPE if any(trimmed for _, _, trimmed in prepared) else None,
            },
        }
        (output / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
        print(f"Evaluation saved: {output}")
        return results

    def evaluate_policy(self, detector: Any, policy: Callable[..., Any], *,
                        base: str, limit: int = 10, jev_model: str | None = None,
                        client: Any = None) -> dict[str, Any]:
        """Paired evaluation: score a frozen detector, then apply a label-free policy.

        Only categorical policy metrics are reported; never invent policy EER/AUC.
        API calls are capped by default and cached persistently under data/cache.
        """
        import hashlib
        import time
        import pandas as pd
        from audio_df.helpers.metrics import compute_eer
        from audio_df.modules.jev import detector_case

        self.validate_testsets()
        if limit <= 0:
            raise ValueError("limit must be positive (external API calls incur cost)")
        artifact = self.model_from_for(base)
        output = self.output
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *detector.cache_namespace)
        print(f"\n[{self.name.upper()}] POLICY EVALUATE | base={base} | artifact={artifact}", flush=True)
        print(f"  API limit per test set: {limit} clips | output: {output}", flush=True)
        scored = {}
        samples = {}
        for name in [self.threshold_from, *self.testsets]:
            df = self.manifest(name)
            total = len(df)
            # Calibration remains complete. Sample test sets before extraction.
            if name != self.threshold_from:
                count = min(limit, total)
                # Independent, stable seeded sampling for each split.
                split_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
                df = df.sample(n=count, random_state=(self.seed + split_seed) % (2**32)).reset_index(drop=True)
                samples[name] = [str(value) for value in df["utt_id"].tolist()]
            trimmed = self._is_trimmed(name)
            print(f"  {name}: {len(df):,}/{total:,} clips | {'trimmed' if trimmed else 'raw'}" +
                  (" | full threshold calibration" if name == self.threshold_from else " | seeded sample"), flush=True)
            if trimmed:
                df = trimmed_manifest(df, self.root / "data" / "cache", name)
            scored[name] = detector.score_manifest(df, output / "scores" / f"{name}.csv", split=name, cache_dir=cache_dir)
        calibration_eer, threshold = compute_eer(scored[self.threshold_from].score, scored[self.threshold_from].label, higher_is_spoof=False)
        (output / "calibration.json").write_text(json.dumps({
            "source": self.threshold_from, "eer": float(calibration_eer),
            "eer_threshold": float(threshold), "n": len(scored[self.threshold_from]),
            "preprocessing": "trimmed" if self._is_trimmed(self.threshold_from) else "raw",
        }, indent=2), encoding="utf-8")
        selected_model = jev_model or __import__("os").getenv("JEV_MODEL", "jev-latest")
        cache = self.root / "data" / "cache" / "jev" / "decisions"
        cache.mkdir(parents=True, exist_ok=True)
        reports = {}
        jev_dir = output / "jev"
        jev_dir.mkdir(parents=True, exist_ok=True)
        with (jev_dir / "sampling.json").open("w", encoding="utf-8") as f:
            json.dump({"method": "pandas.DataFrame.sample", "seed": self.seed,
                       "limit_per_testset": limit, "selected_utt_ids": samples,
                       "calibration_source": self.threshold_from, "calibration": "full dataset for threshold calibration"}, f, indent=2)
        requests_file = jev_dir / "requests.jsonl"
        responses_file = jev_dir / "responses.jsonl"
        for name, table in scored.items():
            if name == self.threshold_from:
                continue
            records = []
            for row in table.itertuples(index=False):
                score = float(row.score)
                base_label = "bonafide" if score >= threshold else "spoof"
                case = detector_case(score, threshold)
                # Keep clip identifiers and ground truth out of API evidence. Cache
                # identity additionally includes the trained detector provenance.
                key = hashlib.sha256(json.dumps({"context": dict(case.context), "prompt": case.prompt,
                    "questions": case.questions, "model": selected_model,
                    "base_architecture": base, "base_artifact": str(artifact.resolve())},
                    sort_keys=True).encode()).hexdigest()
                target = cache / f"{key}.json"
                cached = target.is_file()
                if cached:
                    response = json.loads(target.read_text())
                    latency = 0.0
                else:
                    started = time.monotonic()
                    raw = policy(case, model=selected_model, **({"client": client} if client is not None else {}))
                    latency = time.monotonic() - started
                    response = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
                    if not isinstance(response, dict):
                        raise TypeError("JEV response must be a dictionary or Pydantic model")
                    temp = target.with_suffix(".tmp")
                    temp.write_text(json.dumps(response, indent=2, default=str))
                    temp.replace(target)
                # Research records are independent of the persistent API cache.
                # Ground-truth and identifiers are stored locally, never sent to JEV.
                original = {col: getattr(row, col) for col in table.columns if hasattr(row, col)}
                original = json.loads(json.dumps(original, default=str))
                request_record = {"split": name, "clip_id": str(row.clip_id),
                    "request_hash": key, "model_requested": selected_model,
                    "state": {"context": dict(case.context), "decision_guidance": case.prompt},
                    "questions": case.questions, "original_detector_response": original}
                response_record = {"split": name, "clip_id": str(row.clip_id),
                    "request_hash": key, "cached": cached, "latency_s": latency,
                    "raw_jev_response": response}
                with requests_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(request_record, default=str) + "\n")
                with responses_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(response_record, default=str) + "\n")
                answer = response.get("answers", {}).get("decision", {})
                verdict = answer.get("choice")
                if verdict not in {"bonafide", "spoof", "uncertain"}:
                    raise ValueError(f"Invalid JEV decision: {verdict!r}")
                records.append({"clip_id": str(row.clip_id), "label": str(row.label),
                    "score": score, "base": base_label, "jev": verdict,
                    "request_hash": key, "latency_s": latency})
            paired = pd.DataFrame(records)
            paired.to_csv(output / f"paired_{name}.csv", index=False)
            decided = paired.jev != "uncertain"
            base_ok = paired.base == paired.label
            policy_ok = paired.jev == paired.label
            reports[name] = {"n": len(paired), "base_accuracy": float(base_ok.mean()),
                "coverage": float(decided.mean()),
                "policy_accuracy_on_decided": float(policy_ok[decided].mean()) if decided.any() else None,
                "corrected_errors": int(((~base_ok) & policy_ok & decided).sum()),
                "introduced_errors": int((base_ok & (~policy_ok) & decided).sum()),
                "abstentions": int((~decided).sum()), "request_latency_s": float(paired.latency_s.sum()),
                "note": "Limited subset is an API demonstration, not a full benchmark. No policy EER/AUC/minDCF."}
        (output / "metrics.json").write_text(json.dumps(reports, indent=2))
        (output / "config.yaml").write_text(yaml.safe_dump({"architecture": self.name, "base": base,
            "model_artifact": self._project_path(artifact), "jev_model_requested": selected_model,
            "testsets": self.testsets, "limit": limit, "seed": self.seed,
            "split_roles": {name: ("validation_threshold" if name == self.threshold_from else "held_out_test")
                            for name in [self.threshold_from, *self.testsets]},
            "policy_evaluated_testsets": list(self.testsets),
            "threshold_source": self.threshold_from,
            "sampling": "seeded random without replacement, before preprocessing",
            "dev_threshold": float(threshold),
            "preprocessing": {name: ("trimmed" if self._is_trimmed(name) else "raw")
                              for name in [self.threshold_from, *self.testsets]}}, sort_keys=False))
        print(f"Policy evaluation saved: {output}", flush=True)
        return reports
