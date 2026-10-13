"""Data pipeline for the tube detector: sampling, seed labels, augmentation, training.

Everything here runs with numpy + OpenCV only. Heavy optional dependencies
(ultralytics, Segment Anything) are imported lazily inside the functions that
need them, so the package imports and its tests run on a machine with no GPU
and no data. See TS-001 §7.
"""
