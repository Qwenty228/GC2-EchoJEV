from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from tqdm.auto import tqdm


def _extract(task):
    path, extractor = task
    return extractor(path)


def parallel_map(extractor, paths, workers=4, desc="Extracting"):
    paths = list(paths)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from tqdm(
            pool.map(
                _extract,
                zip(paths, repeat(extractor)),
                chunksize=16,
            ),
            total=len(paths),
            desc=desc,
            unit="clip",
        )