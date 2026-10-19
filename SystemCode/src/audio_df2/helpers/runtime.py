import argparse
import inspect
from functools import wraps
from pathlib import Path


class Run:
    def __init__(self, args):
        self.args = args
        self.root = Path.cwd()
        self.input = args.i
        self.output = Path(args.o)
        self.output.mkdir(parents=True, exist_ok=True)

    @classmethod
    def cli(cls, dataset="2019LA"):
        def decorator(func):
            sig = inspect.signature(func)

            @wraps(func)
            def wrapper():
                parser = argparse.ArgumentParser(
                    description=func.__doc__
                )

                parser.add_argument("--i", default=dataset)
                parser.add_argument("--o", default="runs")
                parser.add_argument("--device", default="auto")
                parser.add_argument("--seed", type=int, default=1337)

                # Automatically expose function parameters as CLI options
                for name, param in sig.parameters.items():
                    if name == "run":
                        continue

                    if param.default is inspect.Parameter.empty:
                        continue

                    parser.add_argument(
                        f"--{name.replace('_', '-')}",
                        type=type(param.default),
                        default=param.default,
                    )

                args = parser.parse_args()
                run = cls(args)

                kwargs = {
                    name: getattr(args, name)
                    for name in sig.parameters
                    if name != "run"
                }

                return func(run, **kwargs)

            return wrapper

        return decorator