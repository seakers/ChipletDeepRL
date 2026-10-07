"""
Shows that the overlap test before commit ab59dbe (Aug 6 2026) misses overlaps.
Loads separating_axis_test from a given commit and checks boxes that clearly overlap.
usage: python experiments/probes/old_overlap_test.py c147c02 1a8656b 05eb825 ab59dbe
"""
import subprocess
import sys
import textwrap

import numpy as np


def load_sat(commit):
    src = subprocess.run(['git', 'show', f'{commit}:utils/evaluation.py'], capture_output=True, text=True,
                         check=True).stdout
    start = src.index('    def separating_axis_test')
    end = src.index('\n    def ', start + 10)
    ns = {'np': np}
    exec(textwrap.dedent(src[start:end]), ns)
    return ns['separating_axis_test']


d = [.3, .2, .15]
I = np.eye(3)
c, s = np.cos(.3), np.sin(.3)
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
cases = [('identical boxes, same place', d, [0, 0, 0], I),
         ('shifted 10 cm', d, [.1, .05, 0], I),
         ('rotated 17 deg about z', d, [.05, 0, 0], Rz),
         ('1 m apart (no overlap)', d, [1, 0, 0], I)]
for commit in sys.argv[1:]:
    sat = load_sat(commit)
    print(commit, ', '.join(f'{name}: {sat(None, d, [0, 0, 0], I, d2, loc, R)}' for name, d2, loc, R in cases))
