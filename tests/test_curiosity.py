import torch
from autogameplayer.core.curiosity import CuriosityEngine, EpisodicMemory
from autogameplayer.core.rnd import RandomNetworkDistillation


def test_episodic_memory_decay():
    memory = EpisodicMemory(k=3)

    # Dummy embedding
    state1 = torch.ones(10)

    # 1st visit (empty buffer) -> max reward (1.0 default for warming up before k steps)
    r1 = memory.compute_reward(state1)
    assert r1 == 1.0
    memory.add(state1)

    # Add a few more slightly different states to fill up buffer
    memory.add(state1 + 0.01)
    memory.add(state1 - 0.01)

    # Now that we have k=3, it should compute actual distance
    r2 = memory.compute_reward(state1)
    memory.add(state1)

    # Third exact visit (should be much lower reward than an unseen state)
    r3 = memory.compute_reward(state1)

    # A totally novel state should have higher reward
    state_novel = torch.ones(10) * 10
    r_novel = memory.compute_reward(state_novel)

    assert r2 > r3, "Reward should drop as state is seen more times in the episode"
    assert r_novel > r3, "Novel state should yield higher reward than familiar state"


def test_rnd_burn_in():
    rnd = RandomNetworkDistillation(input_dim=10, burn_in_steps=3)

    obs = torch.randn(2, 10)

    # Burn in 0
    r0 = rnd.compute_intrinsic_reward(obs, train=True)
    assert r0 == 0.0, "Reward should be 0 during burn-in"
    assert rnd.steps_seen == 2

    # Burn in 1 (now we hit 4 total steps explored, passing burn-in of 3)
    r1 = rnd.compute_intrinsic_reward(obs, train=True)
    assert rnd.steps_seen == 4
    assert r1 > 0.0, "Reward should be >0 after burn-in"


def test_combined_intrinsic_formulation():
    engine = CuriosityEngine()

    # Dummy high episodic, low lifelong (e.g. newly entered an old room during this episode)
    r_episodic_only = engine.compute_intrinsic_reward(torch.randn(10), r_lifelong=0.5)

    # Dummy high episodic, high lifelong (e.g. newly discovered room entirely)
    r_both = engine.compute_intrinsic_reward(torch.randn(10), r_lifelong=2.0)

    # The lifelong bonus scales the episodic reward
    assert r_both > r_episodic_only
