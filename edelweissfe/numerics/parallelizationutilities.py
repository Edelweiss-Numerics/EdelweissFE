#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import os
import sys


def isFreeThreadingSupported() -> bool:
    """Check if free threading is supported in the current build of EdelweissFE.

    Free threading allows for parallel computations using multiple threads.
    This function checks if the current build of EdelweissFE supports this feature.

    Returns:
        bool: True if free threading is supported, False otherwise.
    """

    return not sys._is_gil_enabled()


def getNumberOfThreads() -> int:
    """Get the number of threads available for parallel computations.

    EdelweissFE has a built-in mechanism to determine the number of threads to be used for parallel computations.
    It checks if the environment variable `OMP_NUM_THREADS` is set. If it is
    and can be converted to an integer, that value is used. If not, the function falls back to using a default value of 1.

    Returns:
        int: Number of threads to be used.
    """

    try:
        env_threads = os.environ.get("OMP_NUM_THREADS")
        num_workers = int(env_threads) if env_threads else 1
    except ValueError:
        num_workers = 1

    return num_workers
