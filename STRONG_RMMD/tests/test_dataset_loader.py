from strong_rmmd.config import MACHINES, CDF_VARIABLES, get_cdf_variable_map


def test_machine_registry_has_six_entries():
    assert len(MACHINES) == 6
    for name in ['NSTX', 'HL2A', 'D3D', 'KSTR', 'EAST', 'ITER']:
        assert name in MACHINES


def test_cdf_variable_map_has_six_machines():
    assert set(CDF_VARIABLES.keys()) == {'nstx', 'hl2a', 'd3d', 'kstr', 'east', 'iter'}
    assert get_cdf_variable_map('NSTX')['n_e'] == 'NE'
