# E-20261004-e2-a1-raw-mfcc-s3337

created: 2026-10-04T11:12:39.784080+00:00
- A1 uses whole utterances (official B02 baseline), not this project's usual 4s crop (configs/data.yaml: audio.crop_seconds).
- train-frame cache: data/cache/a1/4172ff00644b8447 (gitignored, resumable)
- train frames: bonafide=873016, spoof=7809362 (dim=60)
- bona fide GMM final avg log-lik=-35.216, spoof GMM final avg log-lik=-32.306
