# experiments

## Setup

```
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe numpy==2.2.5 scipy==1.15.2 matplotlib==3.10.1 pandas==2.2.3 pygad==3.5.0 pymoo==0.6.1.5 dash==4.4.1 plotly==6.9.0 tqdm
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
```

pygad and pymoo match `requirements.txt`. torch uses cu128 because the cu126 build in
`requirements.txt` doesn't support RTX 50 series cards.

## Running

```
python experiments/build_reference.py --n 16000
python experiments/runner.py --tag test --methods ga,ppo,repair --seeds 0-7 --budget 64000 --ref experiments/reference/provisional.npy
```

Methods: `random`, `ga`, `ppo`, `repair`, `wsga`, `imga`, `wsimga`. The hybrids need
`--actors DIR` with the trained `.pth` files. `--set key=value` overrides a param for every
job (for example `--set legacy_ppo=true`).

To run in parallel, use `--shard k --nshards N` with `OMP_NUM_THREADS=1`. Results go to
`data/<tag>_shard_<k>.jsonl`. If a job fails, its error is saved in that file; delete the line to rerun it.

`reference/provisional.npy` is built from 16,000 seeded random designs with the same rule as
`random_search.py`. Results on it can't be compared to results that used the lab's `max_values.npy`.
