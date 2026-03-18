import pytest
import torch
import numpy as np
from autogameplayer.rewards.normalizer import RewardNormalizer
from autogameplayer.core.bandit import SlidingWindowUCB
from autogameplayer.core.context import AgentContext
from autogameplayer.muzero.trainer import transform_value, inverse_transform_value
from autogameplayer.core.models import Observation, GameState


def test_reward_normalizer_clipping():
    norm = RewardNormalizer(curiosity_burst_threshold=1.5)
    
    # Simple observation mock
    obs = Observation(state=GameState(vision_delta=0.1, vision_vector=np.zeros(384)))
    
    # High reward should be clipped
    r = norm.normalize(10.0, obs)
    assert r == 1.5
    assert obs.state.context["curiosity_burst"] is True
    
    # Low reward preserved
    r = norm.normalize(0.5, obs)
    assert r == 0.5
    assert obs.state.context["curiosity_burst"] is False


@pytest.mark.asyncio
async def test_bandit_warm_start():
    bandit = SlidingWindowUCB(num_arms=4, window_size=10)
    
    # Should select each arm once initially
    selected = []
    for _ in range(4):
        arm = await bandit.select_arm()
        selected.append(arm.id)
    
    assert set(selected) == {0, 1, 2, 3}


@pytest.mark.asyncio
async def test_bandit_convergence():
    bandit = SlidingWindowUCB(num_arms=2, window_size=100, exploration_constant=0.1)
    
    # Arm 1 is much better
    for _ in range(50):
        arm = await bandit.select_arm()
        reward = 10.0 if arm.id == 1 else 1.0
        await bandit.report_episode_result(arm.id, reward)
        
    # Should now prefer arm 1
    recent_pulls = []
    for _ in range(10):
        arm = await bandit.select_arm()
        recent_pulls.append(arm.id)
        
    assert recent_pulls.count(1) > recent_pulls.count(0)


def test_bellman_transform_roundtrip():
    x = torch.tensor([0.5, 10.0, -2.0, 100.0])
    transformed = transform_value(x)
    recovered = inverse_transform_value(transformed)
    
    # Should be close to original
    assert torch.allclose(x, recovered, atol=1e-5)


def test_agent_context_stagnation():
    ctx = AgentContext()
    assert ctx.consecutive_stuck_steps == 0
    assert ctx.is_stuck is False
    
    ctx.consecutive_stuck_steps = 25
    ctx.is_stuck = True
    assert ctx.is_stuck is True
