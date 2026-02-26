"""Generate test fixture data for leaflet analyzer tests.

This script runs LeafletAssigner on topology/trajectory data, then
instantiates Analyzer to extract expected assertion values.  The results
are saved as pickle files that the pytest suite loads as fixtures.

Usage
-----
Default — use the small ``example/`` data shipped with the repository::

    uv run python test/generate_test_data.py

Custom — use full data on the Shanghai server::

    uv run python test/generate_test_data.py \\
        --top  /share/home/qjxu/data/plasma@martini/asy/Traj-0/md-dry.gro \\
        --traj /share/home/qjxu/data/plasma@martini/asy/Traj-0/step7_production_skip100.xtc \\
        --start 0 --stop 100

Override the output directory::

    uv run python test/generate_test_data.py --output-dir /tmp/test_fixtures
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io import write_file
from src.leaflet_assigner import LeafletAssigner

# --- Configuration defaults ---
DEFAULT_TOP_FILE = "example/dry.gro"
DEFAULT_TRAJ_FILE = "example/short_traj.xtc"
DEFAULT_OUTPUT_DIR = "test/example/analyzer"
DEFAULT_START = 0
DEFAULT_STOP = 2

HEAD_ATOM = (
    "APC:GL2|BNSM:AM1 AM2|DAPC:GL1 GL2|DAPE:GL1 GL2|DAPS:GL1 GL2|DBSM:AM1 AM2|"
    "DOPC:GL1 GL2|DOPE:GL1 GL2|DPCE:AM1 AM2|DPG1:AM1 AM2|DPG3:AM1 AM2|DPSM:AM1 AM2|"
    "DUPE:GL1 GL2|DUPS:GL1 GL2|DXCE:AM1 AM2|DXG1:AM1 AM2|DXG3:AM1 AM2|DXSM:AM1 AM2|"
    "IPC:GL2|OPC:GL2|PADG:GL1 GL2|PAPA:GL1 GL2|PAPC:GL1 GL2|PAPE:GL1 GL2|"
    "PAPI:GL1 GL2|PAPS:GL1 GL2|PEPC:GL1 GL2|PGSM:AM1 AM2|PIDG:GL1 GL2|"
    "PIPA:GL1 GL2|PIPC:GL1 GL2|PIPE:GL1 GL2|PIPI:GL1 GL2|PIPS:GL1 GL2|"
    "PIPX:GL1 GL2|PNCE:AM1 AM2|PNG1:AM1 AM2|PNG3:AM1 AM2|PNSM:AM1 AM2|"
    "PODG:GL1 GL2|POP1:GL1 GL2|POP2:GL1 GL2|POP3:GL1 GL2|POPA:GL1 GL2|"
    "POPC:GL1 GL2|POPE:GL1 GL2|POPI:GL1 GL2|POPS:GL1 GL2|POPX:GL1 GL2|"
    "POSM:AM1 AM2|PPC:GL2|PQPE:GL1 GL2|PQPS:GL1 GL2|PUDG:GL1 GL2|"
    "PUPA:GL1 GL2|PUPC:GL1 GL2|PUPE:GL1 GL2|PUPI:GL1 GL2|PUPS:GL1 GL2|"
    "UPC:GL2|XNCE:AM1 AM2|XNG1:AM1 AM2|XNG3:AM1 AM2|XNSM:AM1 AM2"
)

TAIL_ATOM = (
    "APC:C5B|BNSM:C4A C6B|DAPC:C5A C5B|DAPE:C5A C5B|DAPS:C5A C5B|DBSM:C4A C5B|"
    "DOPC:C4A C4B|DOPE:C4A C4B|DPCE:C3A C4B|DPG1:C3A C4B|DPG3:C3A C4B|DPSM:C3A C4B|"
    "DUPE:D5A D5B|DUPS:D5A D5B|DXCE:C5A C6B|DXG1:C5A C6B|DXG3:C5A C6B|DXSM:C5A C6B|"
    "IPC:C4B|OPC:C4B|PADG:C5A C4B|PAPA:C5A C4B|PAPC:C5A C4B|PAPE:C5A C4B|"
    "PAPI:C5A C4B|PAPS:C5A C4B|PEPC:C5A C4B|PGSM:C3A C5B|PIDG:C4A C4B|"
    "PIPA:C4A C4B|PIPC:C4A C4B|PIPE:C4A C4B|PIPI:C4A C4B|PIPS:C4A C4B|"
    "PIPX:C4A C4B|PNCE:C3A C6B|PNG1:C3A C6B|PNG3:C3A C6B|PNSM:C3A C6B|"
    "PODG:C4A C4B|POP1:C4A C4B|POP2:C4A C4B|POP3:C4A C4B|POPA:C4A C4B|"
    "POPC:C4A C4B|POPE:C4A C4B|POPI:C4A C4B|POPS:C4A C4B|POPX:C4A C4B|"
    "POSM:C3A C4B|PPC:C4B|PQPE:C5A C4B|PQPS:C5A C4B|PUDG:D5A C4B|"
    "PUPA:D5A C4B|PUPC:D5A C4B|PUPE:D5A C4B|PUPI:D5A C4B|PUPS:D5A C4B|"
    "UPC:D5B|XNCE:C5A C6B|XNG1:C5A C6B|XNG3:C5A C6B|XNSM:C5A C6B"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test fixture pickle files for the leaflet analyzer tests."
    )
    parser.add_argument(
        "--top",
        default=DEFAULT_TOP_FILE,
        help=f"Topology file (default: {DEFAULT_TOP_FILE})",
    )
    parser.add_argument(
        "--traj",
        default=DEFAULT_TRAJ_FILE,
        help=f"Trajectory file (default: {DEFAULT_TRAJ_FILE})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help=f"First frame (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=DEFAULT_STOP,
        help=f"Last frame (exclusive) (default: {DEFAULT_STOP})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for pickle files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    TOP_FILE = args.top
    TRAJ_FILE = args.traj
    START = args.start
    STOP = args.stop
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Run LeafletAssigner ---
    print("=" * 60)
    print("Step 1: Running LeafletAssigner...")
    print("=" * 60)

    assigner = LeafletAssigner(
        topFile=TOP_FILE,
        trajFile=TRAJ_FILE,
        selHeadAtom=HEAD_ATOM,
        selTailAtom=TAIL_ATOM,
        start=START,
        stop=STOP,
        chunkSize=100,
        outputPref=OUTPUT_DIR,
    )
    assigner.run()

    # Rename to match the test fixture naming convention
    src_leaflet = os.path.join(OUTPUT_DIR, "leaflet.pickle")
    src_args = os.path.join(OUTPUT_DIR, "leaflet_args.pickle")
    dst_leaflet = os.path.join(OUTPUT_DIR, "vesicle_leaflet.pickle")
    dst_args = os.path.join(OUTPUT_DIR, "vesicle_leaflet_args.pickle")

    shutil.move(src_leaflet, dst_leaflet)
    shutil.move(src_args, dst_args)

    print(f"\nSaved: {dst_leaflet}")
    print(f"Saved: {dst_args}")

    # --- Step 2: Load through Analyzer to extract expected values ---
    print("\n" + "=" * 60)
    print("Step 2: Loading through Analyzer to extract expected values...")
    print("=" * 60)

    from src.leaflet_analyzer import Analyzer

    analyzer = Analyzer(dst_leaflet, dst_args)

    vesicle_data = {
        "totalResNum": analyzer.totalResNum,
        "notSolventResNum": analyzer.notSolventResNum,
        "selectedRes": analyzer.selectedRes,
        "notSelectedRes": analyzer.notSelectedRes,
        "headAtomIdx_noNan": analyzer.headAtomIdx_noNan,
    }
    print(f"  totalResNum = {analyzer.totalResNum}")
    print(f"  notSolventResNum = {analyzer.notSolventResNum}")

    # Get leaflet location
    analyzer.get_leaflet_location("vesicle", 0)
    location_0_1 = analyzer.leafletCollection[0][(0, 1)].location
    location_0_2 = analyzer.leafletCollection[0][(0, 2)].location
    vesicle_data["location_0_1"] = location_0_1
    vesicle_data["location_0_2"] = location_0_2
    print(f"  location (0,1) = {location_0_1}")
    print(f"  location (0,2) = {location_0_2}")

    # Find unassigned molecules
    unassigned = analyzer.find_unassigned_molecules(0)
    vesicle_data["unAssigned"] = unassigned
    print(f"  unassigned count = {len(unassigned)}")

    # Assign molecules
    analyzer.assign_molecules(0)
    vesicle_data["assignedIdx1"] = analyzer.leafletCollection[0][(0, 1)].molIdx
    vesicle_data["assignedIdx2"] = analyzer.leafletCollection[0][(0, 2)].molIdx
    print(f"  assigned leaflet 1 count = {len(vesicle_data['assignedIdx1'])}")
    print(f"  assigned leaflet 2 count = {len(vesicle_data['assignedIdx2'])}")

    dst_data = os.path.join(OUTPUT_DIR, "vesicle_data.pickle")
    write_file(vesicle_data, dst_data)
    print(f"\nSaved: {dst_data}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Done! Generated test fixture files:")
    print("=" * 60)
    for f in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {filepath}  ({size:,} bytes)")

    # Print values needed for test assertions
    print("\n--- Values for test assertions ---")
    print(f"  analyzer.totalResNum == {analyzer.totalResNum}")
    print(f"  analyzer.notSolventResNum == {int(analyzer.notSolventResNum)}")
    print(f"  location (0,1) ≈ {location_0_1}")
    print(f"  location (0,2) ≈ {location_0_2}")


if __name__ == "__main__":
    main()
