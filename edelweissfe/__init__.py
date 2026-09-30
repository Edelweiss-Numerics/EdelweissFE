#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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

if sys.platform == "win32":
    # Since Python 3.8, Windows resolves the DLLs that extension modules depend on without looking at PATH. Make
    # Marmot.dll, which the Marmot-backed extensions link against, findable: in MARMOT_INSTALL_DIR if set (as for
    # the build, see setup.py), otherwise in the environment's Library prefix, where conda installs it.
    _marmotDllDir = os.path.join(os.environ.get("MARMOT_INSTALL_DIR", os.path.join(sys.prefix, "Library")), "bin")
    if os.path.isdir(_marmotDllDir):
        os.add_dll_directory(_marmotDllDir)
