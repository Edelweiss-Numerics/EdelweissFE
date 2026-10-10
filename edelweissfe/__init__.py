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
import pathlib
import sys

if sys.platform == "win32":
    # Since Python 3.8, Windows resolves the DLLs that extension modules depend on without looking at PATH. Make
    # Marmot.dll findable for the Marmot-backed extensions: from the Marmot installation they were built against,
    # recorded by setup.py -- not from the runtime environment, which may hold a different Marmot build.
    _marmotInstallDirFile = pathlib.Path(__file__).parent / "marmot_install_dir.txt"
    if _marmotInstallDirFile.is_file():
        _marmotDllDir = pathlib.Path(_marmotInstallDirFile.read_text(encoding="utf-8").strip()) / "bin"
        if _marmotDllDir.is_dir():
            os.add_dll_directory(str(_marmotDllDir))
