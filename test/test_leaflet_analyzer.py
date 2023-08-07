import numpy as np
from src.leaflet_analyzer import Analyzer
from src.io import read_file

def test_leaflet_analyzer_initialization(set_global_data):

    vesAnalyzer = Analyzer('test/example/analyzer/vesicle_leaflet.pickle', 
                           'test/example/analyzer/vesicle_leaflet_args.pickle')

    assert vesAnalyzer.totalResNum == 6664
    assert vesAnalyzer.notSolventResNum == 4659
    vesicleData = read_file('test/example/analyzer/vesicle_data.pickle')
    assert (vesAnalyzer.selectedRes == vesicleData['selectedRes']).all()
    assert (vesAnalyzer.notSelectedRes == vesicleData['notSelectedRes']).all()
    assert (vesAnalyzer.headAtomIdx_noNan == vesicleData['headAtomIdx_noNan']).all()

    #vesAnalyzer.load_traj()
    set_global_data('vesAnalyzer', vesAnalyzer)
    set_global_data('vesicleData', vesicleData)

def test_get_leaflet_location(get_global_data):

    vesAnalyzer = get_global_data('vesAnalyzer')
    vesAnalyzer.get_leaflet_location('vesicle', 0)
    assert np.isclose(vesAnalyzer.leafletCollection[0][(0, 1)].location, 136.61499)
    assert np.isclose(vesAnalyzer.leafletCollection[0][(0, 2)].location, 104.31151)

def test_find_unassigned_molecules(get_global_data):

    vesAnalyzer = get_global_data('vesAnalyzer')
    unAssigned = vesAnalyzer.find_unassigned_molecules(0)
    vesicleData = get_global_data('vesicleData')
    assert (unAssigned == vesicleData['unAssigned']).all()

def test_assign_molecules(get_global_data):

    vesAnalyzer = get_global_data('vesAnalyzer')
    vesAnalyzer.assign_molecules(0)
    vesicleData = get_global_data('vesicleData')
    assert (vesAnalyzer.leafletCollection[0][(0, 1)].molIdx == vesicleData['assignedIdx1']).all()
    assert (vesAnalyzer.leafletCollection[0][(0, 2)].molIdx == vesicleData['assignedIdx2']).all()

def test_calculate_correlation(get_global_data):

    vesAnalyzer = get_global_data('vesAnalyzer')
    atomIdx = vesAnalyzer.top.select_atoms('resname CHOL and name ROH').ids
    resIdx = vesAnalyzer.top.select_atoms('resname CHOL and name ROH').resids
    print(vesAnalyzer.calculate_correlation(0, 1, 50, atomIdx, resIdx))
    [0.04107462251528567]