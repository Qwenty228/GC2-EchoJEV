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

    @property
    def output(self) -> Path:
        if self._run_dir is None:
            self._run_dir = _new_directory(self.output_root, f"E-{_timestamp()}-{self.name}")
        return self._run_dir

    def manifest(self, split: str):
        return load_manifest(self.root / "data" / "manifests" / f"{split}.csv", self.root)

    def train(self, model: Any) -> Any:
        # Feature caches are reusable and are not stored with evaluation results.
        cache_dir = feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace)
        params = inspect.signature(model.fit).parameters
        extra = {"dev": self.manifest("dev")} if "dev" in params else {}
        return model.fit(self.manifest("train"), cache_dir, seed=self.seed, **extra)

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
        scored = {}
        for name in self.testsets:
            df = self.manifest(name)
            destination = output / "scores" / f"{name}.csv"
            scored[name] = model.score_manifest(
                df, destination, split=name,
                cache_dir=feature_cache_dir(self.root / "data" / "cache", *model.cache_namespace),
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
        }
        (output / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
        print(f"Evaluation saved: {output}")
        return results
