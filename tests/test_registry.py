from rankstudy.registry import build_registry

def test_registry_has_primary_and_exploratory_unique_cells():
    rows=build_registry(); ids=[x.id for x in rows]
    assert len(rows)==43 and len(set(ids))==43
    for x in ('L_FIXED','I_FIXED','ADALORA_REF','L_RAW','I_GATE','T0_GATE','T1_GATE','L_PILOT_I_STUDENT','I_GLOBAL_SVD','L_RAW_UNCERTAINTY','I_GATE_UNCERTAINTY'):
        assert x in ids
    assert {'I_ADALORA_SVD','I_ADALORA_SVD_NORM'} <= set(ids)
