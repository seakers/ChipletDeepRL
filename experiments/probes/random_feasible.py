# Count feasible designs from a commit's own random search. Run with cwd = a worktree of that commit:
#   python random_feasible.py 2000
import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents
try:
    from optimization.random_search import run_random_search
except ImportError:
    from random_search import run_random_search

n = int(sys.argv[1])
comps = getComponents()
comps = comps[0] if isinstance(comps, tuple) else comps
ev = Chiplet_Configuration_Design(comps, StructPanel())
flags = []
orig = ev.evaluate
def rec(x):
    out = orig(x)
    flags.append(bool(out[1]))
    return out
ev.evaluate = rec
t = time.time()
try:
    run_random_search({'num_epochs': n // 16, 'mini_batch_size': 16}, ev)
except Exception as e:
    print('random_search raised after', len(flags), 'evals:', repr(e)[:200])
f = np.array(flags)
print(f'{len(comps)} components, {len(f)} evals, {int((~f).sum())} feasible, {time.time()-t:.0f}s')
