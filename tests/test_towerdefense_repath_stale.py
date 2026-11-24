import pytest
import gameplay.towerdefense_repath_stale as mod

CASES = [
    ("invalidate_inflight", (1,)),
    ("grid_version", (1,)),
    ("target_walkable", (1,)),
    ("claim_order", (1,)),
    ("release_cell", (1,)),
    ("batch_converge", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("repath", (1,)),
    ("metrics", (1,)),
    ("occupancy", (1,)),
    ("path_cache", (1,)),
    ("commit", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
