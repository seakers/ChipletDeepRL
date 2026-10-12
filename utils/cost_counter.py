"""Counts of the work done outside full evaluations, for charging it under different cost models.

pair_tests: box-pair (or box-panel) overlap tests considered, before any bounding-sphere culling.
A full evaluation's constraint check does a fixed number of these, so aux pair tests divided by that
number gives the constraint work in full-design equivalents.
"""
import collections

COUNTS = collections.Counter()


def reset():
    COUNTS.clear()
