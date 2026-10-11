# Run the IDETC commit's own methods (opt_main.py at 110a859) and record every evaluated design.
# Run with cwd = a worktree of 110a859:
#   python run_idetc.py SEED OUTDIR [--epochs 500] [--repair-epochs 500] [--date-str FOLDER] [--methods "Random Search,..."]
import os, sys, time, random, argparse, datetime, pickle
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('seed', type=int)
ap.add_argument('out')
ap.add_argument('--epochs', type=int, default=500)
ap.add_argument('--methods', default='Random Search,Genetic Algorithm,RL Informed Env,Warm Start GA,Design Repair,Intelligent Mutation GA,Warm Start Intelligent Mutation GA')
# design repair stops after num_epochs * 64 designs but evaluates every edit (~17 evaluations per design)
ap.add_argument('--repair-epochs', type=int, default=500)
# reuse an earlier run's results folder, so the warm-start methods find that run's trained PPO policy
ap.add_argument('--date-str')
a = ap.parse_args()

import opt_main
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents

if not torch.cuda.is_available():
    # the commit autocasts to fp16, which CPU torch can't sample a Dirichlet in; run CPU in fp32
    import transformer_architecture_informed_state as T
    T.autocast = lambda device_type, dtype: torch.autocast(device_type, enabled=False)

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(2)

# params block from opt_main.main() at 110a859
params = {
    'num_epochs': a.epochs,
    'mini_batch_size': 64,
    'gamma': 0.999,
    'lambda': 0.95,
    'learning_rate': 0.0001,
    'clip_ratio': 0.2,
    'update_iterations': 5,
    'target_kl': 0.003,
    'date_str': a.date_str or f'idetc_s{a.seed}_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    'model_folder': None,
}
comps, _ = getComponents()
ev = Chiplet_Configuration_Design(comps, StructPanel())
os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
os.makedirs(a.out, exist_ok=True)

current = {'name': None}
rec = {}
orig = ev.evaluate
def recorded(x):
    out = orig(x)
    # keep only the designs this commit calls feasible (design repair evaluates ~1M designs)
    r = rec.setdefault(current['name'], {'x': [], 'obj': [], 'n': 0, 'first': []})
    r['n'] += 1
    if not out[1]:
        r['x'].append(np.array(x, dtype=float)); r['obj'].append(np.array(out[0], dtype=float)); r['first'].append(r['n'])
    return out
ev.evaluate = recorded

wanted = [m.strip() for m in a.methods.split(',')]
max_values = None
for method in opt_main.initialize_methods():
    if method.name not in wanted:
        continue
    current['name'] = method.name
    t = time.time()
    print(f'== {method.name} seed {a.seed}', flush=True)
    params['num_epochs'] = a.repair_epochs if method.name == 'Design Repair' else a.epochs
    try:
        results, mv = opt_main.run_single_method(method, params, ev, max_values)
        if method.name == 'Random Search' and mv is not None:
            max_values = mv
    except Exception as e:
        print(f'{method.name} raised: {e!r}'[:500], flush=True)
    r = rec.get(method.name, {'x': [], 'obj': [], 'n': 0, 'first': []})
    print(f'== {method.name}: {r["n"]} evals, {len(r["x"])} feasible by the commit test, {time.time()-t:.0f}s', flush=True)
    key = method.name.replace(' ', '_').lower()
    np.savez_compressed(os.path.join(a.out, f'{key}_s{a.seed}.npz'), x=np.array(r['x']), obj=np.array(r['obj']),
                        n=r['n'], first=np.array(r['first']), max_values=np.array(max_values if max_values is not None else []))
