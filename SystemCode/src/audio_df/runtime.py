"""Small CLI runtime: model artifacts are separate from evaluation runs."""
from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from collections.abc import Callable
import yaml

from audio_df.data.manifests import load_manifest
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


class Run:
    def __init__(
        self, *, name: str, dataset: str = "2019LA", output: str = "runs",
        root: str = ".", seed: int = 1337, testsets: str = "dev,eval",
        model_from: str | None = None, command: str = "run",
    ) -> None:
        if dataset.lower() not in {"2019la", "2019", "asvspoof2019"}:
            raise ValueError("Only 2019LA is currently supported")
        if command not in {"train", "evaluate", "run"}:
            raise ValueError(f"Unknown command: {command}")
        self.name = name
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.output_root = (self.root / output).resolve()
        self.artifact_root = self.root / "artifacts" / name
        self.seed = seed
        self.testsets = [part.strip() for part in testsets.split(",") if part.strip()]
        self.command = command
        self._model_from_arg = model_from
        self._artifact_dir: Path | None = None
        self._run_dir: Path | None = None

    @classmethod
    def cli(cls, dataset: str = "2019LA") -> Callable[..., Any]:
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            # Derive from the decorated function's defining file, not call stack.
            name = Path(inspect.getfile(func)).stem
            signature = inspect.signature(func)
            manifest_dir = Path("data/manifests")
            available = sorted(p.stem for p in manifest_dir.glob("*.csv")  if p.stem != "train")
            

            @wraps(func)
            def invoke(argv: list[str] | None = None) -> Any:
                
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
                    help=f"Training dataset identifier (default: {dataset})",
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
                    default="dev,eval",
                    metavar="NAMES",
                    help=(
                            "Comma-separated test sets (default: dev,eval). "
                        f"Available: {', '.join(available) or 'none'}"
                    )
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
            return chosen
        if not self.artifact_root.exists():
            raise FileNotFoundError(f"No {self.name} model found. Train first.")
        valid = sorted(p for p in self.artifact_root.iterdir()
                       if p.is_dir() and (p / "COMPLETE").is_file())
        if not valid:
            raise FileNotFoundError(f"No completed {self.name} model found. Train first.")
        return valid[-1]

    def model_from_for(self, architecture: str) -> Path:
        """Resolve a completed artifact of another architecture, respecting --model-from."""
        if not architecture or not architecture.replace("_", "").isalnum():
            raise ValueError("Invalid architecture")
        if self._model_from_arg:
            selected = (self.root / self._model_from_arg).resolve()
            if not (selected / "COMPLETE").is_file():
                raise FileNotFoundError(f"Model artifact is not complete: {selected}")
            if selected.parent != (self.root / "artifacts" / architecture).resolve():
                raise ValueError(f"--model-from must select an {architecture} artifact")
            return selected
        parent = self.root / "artifacts" / architecture
        valid = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "COMPLETE").is_file()) if parent.is_dir() else []
        if not valid:
            raise FileNotFoundError(f"No completed {architecture} artifact found")
        return valid[-1]

    @property
    def output(self) -> Path:
        if self._run_dir is None:
            self._run_dir = _new_directory(self.output_root, f"E-{_timestamp()}-{self.name}")
        return self._run_dir

    def manifest(self, split: str):
        return load_manifest(self.root / "data" / "manifests" / f"{split}.csv", self.root)

    def train(self, model: Any) -> Any:
        # Training and early-stopping development data are always raw.
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace)
        train_df = self.manifest("train")
        params = inspect.signature(model.fit).parameters
        extra = {"dev": self.manifest("dev")} if "dev" in params else {}
        print(f"\n[{self.name.upper()}] TRAIN | dataset={self.dataset} | seed={self.seed}", flush=True)
        print(f"  train: {len(train_df):,} clips | raw", flush=True)
        if "dev" in extra:
            print(f"  dev:   {len(extra['dev']):,} clips | raw (training validation)", flush=True)
        print(f"  feature cache: {cache_dir}", flush=True)
        print(f"  artifact destination: {self.artifact_root}", flush=True)
        return model.fit(train_df, cache_dir, seed=self.seed, **extra)

    def save(self, model: Any) -> Path:
        directory = _new_directory(self.artifact_root, _timestamp())
        model.save(directory)
        # A completion marker is written LAST. Incomplete artifacts are ignored.
        (directory / "COMPLETE").write_text("ok\n", encoding="utf-8")
        self._artifact_dir = directory
        print(f"Model saved: {directory}")
        return directory
    
    def validate_testsets(self) -> None:
        manifest_dir = self.root / "data" / "manifests"

        available = {p.stem for p in manifest_dir.glob("*.csv")}
        requested = set(self.testsets)

        unknown = requested - available

        if unknown:
            raise ValueError(
                f"Unknown testsets: {sorted(unknown)}. "
                f"Available: {sorted(available)}"
            )

    def evaluate(self, model: Any) -> dict[str, Any]:
        self.validate_testsets()
        if "dev" not in self.testsets:
            raise ValueError("Include dev in --testsets to select the decision threshold")
        artifact = self.model_from
        output = self.output
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace)
        prepared = []
        for name in self.testsets:
            df = self.manifest(name)
            trimmed = self.dataset.lower() in {"2019la", "2019", "asvspoof2019"} and (
                name == "dev" or name == "eval" or name.startswith("eval_")
            )
            prepared.append((name, df, trimmed))
        print(f"\n[{self.name.upper()}] EVALUATE | dataset={self.dataset}", flush=True)
        print(f"  model artifact: {artifact}", flush=True)
        for name, df, trimmed in prepared:
            role = "threshold selection" if name == "dev" else "test"
            print(f"  {name}: {len(df):,} clips | {'trimmed' if trimmed else 'raw'} ({role})", flush=True)
        print(f"  feature cache: {cache_dir}", flush=True)
        print(f"  output: {output}", flush=True)
        scored = {}
        for name, df, trimmed in prepared:
            print(f"[{self.name.upper()}] Scoring {name} ({'trimmed' if trimmed else 'raw'})", flush=True)
            if trimmed:
                df = trimmed_manifest(df, self.root / "data" / "cache", name)
            destination = output / "scores" / f"{name}.csv"
            scored[name] = model.score_manifest(
                df, destination, split=name, cache_dir=cache_dir,
            )
        results = {}
        for name, table in scored.items():
            if name == "dev":
                continue
            results[name] = evaluate_scores(scored["dev"], table, output / f"metrics_{name}.json")
            print(f"{name}: EER={results[name]['eer']:.4%}")
        (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        record = {
            "architecture": self.name,
            "dataset": self.dataset,
            "model_artifact": str(artifact),
            "testsets": self.testsets,
            "seed": self.seed,
            "evaluation_preprocessing": {
                "testsets": {name: ("trimmed" if trimmed else "raw")
                             for name, _, trimmed in prepared},
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
        from audio_df.modules.jev import JEVState

        self.validate_testsets()
        if "dev" not in self.testsets:
            raise ValueError("Include dev in --testsets for threshold calibration")
        if limit <= 0:
            raise ValueError("limit must be positive (external API calls incur cost)")
        artifact = self.model_from_for(base)
        output = self.output
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *detector.cache_namespace)
        print(f"\n[{self.name.upper()}] POLICY EVALUATE | base={base} | artifact={artifact}", flush=True)
        print(f"  API limit per test set: {limit} clips | output: {output}", flush=True)
        scored = {}
        samples = {}
        for name in self.testsets:
            df = self.manifest(name)
            total = len(df)
            # Dev must remain complete for reliable EER threshold calibration.
            # Sample BEFORE trimming or extracting features on all other splits.
            if name != "dev":
                count = min(limit, total)
                # Independent, stable seeded sampling for each split.
                split_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
                df = df.sample(n=count, random_state=(self.seed + split_seed) % (2**32)).reset_index(drop=True)
                samples[name] = [str(value) for value in df["utt_id"].tolist()]
            trimmed = self.dataset.lower() in {"2019la", "2019", "asvspoof2019"} and (name == "dev" or name == "eval" or name.startswith("eval_"))
            print(f"  {name}: {len(df):,}/{total:,} clips | {'trimmed' if trimmed else 'raw'}" +
                  (" | full threshold calibration" if name == "dev" else " | seeded sample"), flush=True)
            if trimmed:
                df = trimmed_manifest(df, self.root / "data" / "cache", name)
            scored[name] = detector.score_manifest(df, output / "scores" / f"{name}.csv", split=name, cache_dir=cache_dir)
        _, threshold = compute_eer(scored["dev"].score, scored["dev"].label, higher_is_spoof=False)
        selected_model = jev_model or __import__("os").getenv("JEV_MODEL", "jev-latest")
        cache = self.root / "data" / "cache" / "jev" / "decisions"
        cache.mkdir(parents=True, exist_ok=True)
        reports = {}
        jev_dir = output / "jev"
        jev_dir.mkdir(parents=True, exist_ok=True)
        with (jev_dir / "sampling.json").open("w", encoding="utf-8") as f:
            json.dump({"method": "pandas.DataFrame.sample", "seed": self.seed,
                       "limit_per_testset": limit, "selected_utt_ids": samples,
                       "dev": "full dataset for threshold calibration"}, f, indent=2)
        requests_file = jev_dir / "requests.jsonl"
        responses_file = jev_dir / "responses.jsonl"
        for name, table in scored.items():
            if name == "dev":
                continue
            records = []
            for row in table.itertuples(index=False):
                score = float(row.score)
                base_label = "bonafide" if score >= threshold else "spoof"
                case = JEVState(
                    context={"base_score": score, "dev_threshold": float(threshold),
                             "base_verdict": base_label, "signed_margin": score - float(threshold),
                             "score_direction": "higher means bonafide"},
                    prompt="Decide whether the supplied detector evidence supports bona fide or spoof. Only one detector score is provided; do not invent acoustic evidence. Abstain when insufficient.",
                    questions={
                        "decision": {"type": "choice", "instructions": "Choose the audio verdict supported by the detector evidence.",
                                     "criteria": {"bonafide": "Evidence favors authentic speech.", "spoof": "Evidence favors spoofed speech.", "uncertain": "Evidence is insufficient for a reliable verdict."}},
                        "evidence_quality": {"type": "score", "instructions": "How sufficient is the supplied evidence?", "criteria": ["Insufficient", "Limited", "Strong"]},
                        "ambiguity": {"type": "noul", "instructions": "Is the supplied evidence ambiguous?"},
                    },
                )
                # The hash excludes ground-truth, identifiers, paths and split names.
                key = hashlib.sha256(json.dumps({"context": dict(case.context), "prompt": case.prompt,
                    "questions": case.questions, "model": selected_model}, sort_keys=True).encode()).hexdigest()
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
            "model_artifact": str(artifact), "jev_model_requested": selected_model,
            "testsets": self.testsets, "limit": limit, "seed": self.seed,
            "sampling": "seeded random without replacement, before preprocessing",
            "dev_threshold": float(threshold),
            "preprocessing": "2019 LA dev/eval trimmed; other datasets raw"}, sort_keys=False))
        print(f"Policy evaluation saved: {output}", flush=True)
        return reports
