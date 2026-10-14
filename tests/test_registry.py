from rankstudy.registry import build_registry

def test_registry_has_all_41_unique_cells():
    rows=build_registry(); ids=[x.id for x in rows]
    assert len(rows)==41 and len(set(ids))==41
    for x in ('L_FIXED','I_FIXED','ADALORA_REF','L_RAW','I_GATE','T0_GATE','T1_GATE','L_PILOT_I_STUDENT','I_GLOBAL_SVD','L_RAW_UNCERTAINTY','I_GATE_UNCERTAINTY'):
        assert x in ids
