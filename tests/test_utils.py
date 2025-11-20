from src.utils import function_timer


def test_timer():
    with function_timer() as timer:
        _ = sum(range(100))
    assert timer.elapsed_time >= 0
