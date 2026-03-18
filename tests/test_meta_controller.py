import pytest
from autogameplayer.core.meta_controller import MetaController
import sqlite3


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    # We don't use the db connection yet, but we'll mock the sqlite3.connect when needed
    yield conn
    conn.close()


def test_sliding_window_ucb_initialization():
    mc = MetaController(window_size=10)
    assert len(mc.window) == 0  # warm start applies to counts/scores, not window
    assert all(c == 1 for c in mc.counts)


@pytest.mark.asyncio
async def test_select_personality_updates_counts():
    mc = MetaController(window_size=10)

    # Selection should return an int within 0-31
    arm = await mc.select_personality()
    assert 0 <= arm < 32

    # However, UCB counts are updated in `update`, not just `select`.
    # Let's test `update` behavior.
    await mc.update(arm, 10.0)

    # Window now has 1 (new)
    assert len(mc.window) == 1
    assert mc.counts[arm] == 2  # 1 warm + 1 actual


@pytest.mark.asyncio
async def test_sliding_window_eviction():
    mc = MetaController(window_size=5)

    assert mc.window.maxlen == 5

    for _ in range(10):
        await mc.update(0, 1.0)

    assert len(mc.window) == 5
    assert mc.counts[0] == 6  # 1 warm + 5 from the sliding window
    assert sum(mc.counts[1:]) == 31  # The other 31 arms were untouched
