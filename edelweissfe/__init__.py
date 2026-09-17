import os as _os

try:
    #: The CPU affinity this process was started with, captured before any submodule --
    #: and so before any OpenMP runtime a compiled extension pulls in -- can narrow it.
    #: ``OMP_PROC_BIND`` makes libgomp pin the *initial* thread to one place the moment it
    #: loads, and every Python worker thread created afterwards inherits that one-core
    #: mask; the thread pool restores this mask in its workers (see
    #: :func:`edelweissfe.numerics.parallelizationutilities.getThreadPool`).
    initialCpuAffinity = frozenset(_os.sched_getaffinity(0))
except (AttributeError, OSError):  # pragma: no cover - non-Linux
    initialCpuAffinity = None
