"""Small reusable experiment runtime and decorator-driven CLI."""
import argparse
import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from audio_df2.data.manifests import load_manifest
from audio_df2.evaluation import evaluate_scores, score_manifest


class Run:
    def __init__(self, *, dataset="2019LA", output="runs/a1_v2", root=".",
                 seed=1337, testsets="dev,eval", model_from=None, command="run"):
        if dataset.lower() not in {"2019la", "2019", "asvspoof2019"}:
            raise ValueError("Only 2019LA is currently supported")
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.testsets = [s.strip() for s in testsets.split(",") if s.strip()]
        self.model_from = Path(model_from) if model_from else self.output / "model"
        self.command = command

    @classmethod
    def cli(cls, dataset="2019LA") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(func):
            signature = inspect.signature(func)

            @wraps(func)
            def invoke(argv=None):
                parser = argparse.ArgumentParser(description=func.__doc__)
                parser.add_argument("command", nargs="?", choices=("run", "train", "evaluate"), default="run")
                parser.add_argument("--i", default=dataset)
                parser.add_argument("--o", default="runs/a1_v2")
                parser.add_argument("--root", default=".")
                parser.add_argument("--seed", type=int, default=1337)
                parser.add_argument("--testsets", default="dev,eval")
                parser.add_argument("--model-from", default=None)
                for name, param in signature.parameters.items():
                    if name == "run" or param.default is inspect.Parameter.empty:
                        continue
                    default = param.default
                    parser.add_argument(f"--{name.replace('_', '-')}",
                                        type=type(default), default=default)
                args = parser.parse_args(argv)
                run = cls(dataset=args.i, output=args.o, root=args.root,
                          seed=args.seed, testsets=args.testsets,
                          model_from=args.model_from, command=args.command)
                kwargs = {name: getattr(args, name) for name in signature.parameters if name != "run"}
                return func(run, **kwargs)

            return invoke
        return decorate

    def manifest(self, split):
        return load_manifest(self.root / "data/manifests" / f"{split}.csv", self.root)

    def train(self, model):
        return model.fit(self.manifest("train"), self.output / "cache", seed=self.seed)

    def save(self, model):
        model.save(self.output / "model")

    def evaluate(self, model):
        if "dev" not in self.testsets:
            raise ValueError("Include dev in --testsets for a fixed dev-selected threshold")
        scored = {
            name: score_manifest(self.manifest(name), model,
                                 self.output / "scores" / f"{name}.csv")
            for name in self.testsets
        }
        results = {}
        for name, table in scored.items():
            if name == "dev":
                continue
            metrics = evaluate_scores(scored["dev"], table,
                                      self.output / f"metrics_{name}.json")
            results[name] = metrics
            print(f"{name}: EER={metrics['eer']:.4%}")
        (self.output / "metrics.json").write_text(json.dumps(results, indent=2))
        return results
