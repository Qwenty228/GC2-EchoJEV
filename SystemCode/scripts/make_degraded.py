#!/usr/bin/env python3
"""Generate controlled ASVspoof 2019 LA evaluation degradations.

Uses the existing seeded, stratified subset from configs/e4_degradations.yaml.
Does not retrain models, download MUSAN, or alter the subset definition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm.auto import tqdm

from audio_df.data.degrade import degrade, clip_seed
from audio_df.data.manifests import stratified_subset
from audio_df.helpers.audio import load_audio


def selected_rows(cfg: dict) -> pd.DataFrame:
    source = pd.read_csv(cfg['subset']['source_manifest'])
    sub = cfg['subset']
    ids = sub.get('clip_ids')
    if ids:
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate clip IDs in subset configuration')
        selected = source.set_index('utt_id', drop=False).reindex(ids)
        if selected['utt_id'].isna().any():
            raise ValueError('Subset contains IDs missing from source manifest')
        return selected.reset_index(drop=True)
    return stratified_subset(source, sub['n'], tuple(sub['strata']), cfg['seed'])


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    df.to_csv(tmp, index=False, lineterminator='\n')
    tmp.replace(path)


def make_clean(cfg: dict, rows: pd.DataFrame) -> Path:
    path = Path(cfg['manifests']['clean_subset'])
    clean = rows.copy()
    clean['condition'] = 'clean'
    write_manifest(clean, path)
    return path


def generate(cfg: dict, rows: pd.DataFrame, name: str, workers: int, overwrite: bool) -> Path:
    spec = cfg['conditions'][name]
    target = Path(cfg['output_dir']) / name
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(cfg['manifests']['out_dir']) / f'{name}.csv'
    noise_paths = []
    if spec['type'] == 'noise':
        noise_dir = Path(spec['noise_dir'])
        if not noise_dir.is_dir():
            raise FileNotFoundError(f'MUSAN noise not found: {noise_dir}. See data/README.md')
        noise_paths = sorted(p for p in noise_dir.rglob('*') if p.suffix.lower() in {'.wav', '.flac'})
        if not noise_paths:
            raise FileNotFoundError(f'No MUSAN noise audio in {noise_dir}')

    def process(record: dict) -> dict:
        clip_id = str(record['utt_id'])
        dest = target / f'{clip_id}.flac'
        if overwrite or not dest.is_file():
            x = load_audio(record['path'], 16000)
            bank = None
            if noise_paths:
                # One deterministic noise clip per utterance. Avoid loading the entire MUSAN bank.
                rng = np.random.default_rng(clip_seed(cfg['seed'], clip_id, name))
                chosen = noise_paths[int(rng.integers(len(noise_paths)))]
                bank = [load_audio(chosen, 16000)]
            y = degrade(x, 16000, name, cfg, cfg['seed'], clip_id, noise_bank=bank)
            tmp = dest.with_suffix('.tmp')
            sf.write(tmp, y, 16000, format='FLAC')
            tmp.replace(dest)
        result = dict(record)
        result.update(path=dest.as_posix(), sample_rate=16000, codec='FLAC', condition=name)
        info = sf.info(str(dest))
        result['duration_s'] = info.frames / info.samplerate
        return result

    records = rows.to_dict('records')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        output = list(tqdm(pool.map(process, records), total=len(records), desc=name, unit='clip'))
    write_manifest(pd.DataFrame(output), manifest_path)
    print(f'Wrote {manifest_path} ({len(output):,} clips)')
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/e4_degradations.yaml'))
    parser.add_argument('--conditions', default='all', help='Comma-separated condition names, or all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--clean-only', action='store_true', help='Only write the clean subset manifest')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    rows = selected_rows(cfg)
    print(f'Selected {len(rows):,} clips from {cfg["subset"]["source_manifest"]}')
    print(f'Clean reference: {make_clean(cfg, rows)}')
    if args.clean_only:
        return
    names = list(cfg['conditions']) if args.conditions == 'all' else args.conditions.split(',')
    for name in names:
        if name not in cfg['conditions']:
            parser.error(f'Unknown condition {name!r}. Available: {list(cfg["conditions"])}')
        generate(cfg, rows, name, args.workers, args.overwrite)


if __name__ == '__main__':
    main()
