#!/usr/bin/env python3
"""Rewrite an EdelweissFE Ensight Gold case so the geometry is given per variable time step.

WHY THIS EXISTS
---------------
With h-adaptivity the mesh changes only on the increments where a refinement actually happens, so the
writer emits one ``geometry.geo_*`` per mesh state and puts them on their own (short) time set, while
the variables sit on a much longer one. That is legal Ensight Gold and it is not lossy -- the refined
mesh IS in the output -- but a reader then has to pair a 3-step geometry set with, say, a 146-step
variable set by time value, and ParaView's Ensight Gold reader does not do that reliably. The usual
symptom is that every frame appears to show the ORIGINAL mesh, which reads as "the refinement never
happened".

This script materialises the pairing explicitly: for every variable step it links the geometry that
was in force at that time, and writes a case file whose geometry uses the variable time set. Nothing
is recomputed and no simulation data is touched -- only symlinks and a new .case file are created, so
it is safe to run against a job that is still writing, and safe to re-run afterwards to pick up the
frames that have appeared since.

USAGE
-----
    python ensight_perstep_geometry.py esExport.case
    -> writes esExport_perstep.case, open THAT one.
"""

import os
import re
import sys


def parseCase(path):
    """Return (timeSets, geometrySet, geometryPattern, variableLines) from an Ensight Gold case file."""
    with open(path) as f:
        text = f.read()

    timeSets = {}
    for block in re.finditer(
        r"time set:\s*(\d+)[^\n]*\n(.*?)(?=time set:|\nFILE|\nGEOMETRY|\nVARIABLE|\Z)", text, re.S
    ):
        number, body = int(block.group(1)), block.group(2)
        values = [float(v) for v in re.findall(r"[-+0-9.eE]+", body.split("time values:", 1)[1])]
        timeSets[number] = values

    geometry = re.search(r"model:\s*(\d+)\s+(\S+)", text)
    variables = re.findall(r"^((?:vector|scalar|tensor) per (?:node|element):.*)$", text, re.M)
    return timeSets, int(geometry.group(1)), geometry.group(2), variables


def main(casePath):
    timeSets, geometrySet, geometryPattern, variableLines = parseCase(casePath)

    variableSets = {int(m.group(1)) for line in variableLines for m in [re.match(r".*?:\s*(\d+)\s", line)]}
    assert len(variableSets) == 1, f"expected one variable time set, found {sorted(variableSets)}"
    variableSet = variableSets.pop()

    geometryTimes, variableTimes = timeSets[geometrySet], timeSets[variableSet]
    print(f"  geometry time set {geometrySet}: {len(geometryTimes)} step(s)")
    print(f"  variable time set {variableSet}: {len(variableTimes)} step(s)")

    root = os.path.dirname(os.path.abspath(casePath))
    stem = geometryPattern.replace("_****", "")  # e.g. esExport/geometry.geo
    linkStem = stem.replace(".geo", "_perstep.geo")

    # For each variable step, the geometry in force is the last one written at or before that time.
    # Strictly "at or before": a geometry written AT a step's time is the mesh that step was solved on.
    linked = 0
    for k, t in enumerate(variableTimes):
        g = max((i for i, gt in enumerate(geometryTimes) if gt <= t + 1e-30), default=0)
        target = f"{os.path.basename(stem)}_{g:04d}"
        link = os.path.join(root, f"{linkStem}_{k:04d}")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
        linked += 1
    print(f"  linked {linked} per-step geometry file(s) -> {linkStem}_****")

    out = os.path.splitext(casePath)[0] + "_perstep.case"
    with open(out, "w") as f:
        f.write("FORMAT\ntype: ensight gold\n\nTIME\n")
        f.write(f"time set: {variableSet} no description\n")
        f.write(f"number of steps: {len(variableTimes)}\n")
        f.write("filename start number: 0\nfilename increment: 1\ntime values: ")
        f.write("\n".join(f"{t:.8e}" for t in variableTimes))
        f.write(f"\n\nGEOMETRY\nmodel: {variableSet} {linkStem}_****\n\nVARIABLE\n")
        f.write("\n".join(variableLines))
        f.write("\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "esExport.case")
