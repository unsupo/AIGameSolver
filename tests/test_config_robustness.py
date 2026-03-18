from autogameplayer.core.config_loader import GameConfig, GameHeuristics, GameProfile, RAMLayout, RewardConfig
from autogameplayer.core.curriculum import Checkpoint

def test_heuristics_extra_fields():
    """Verify that GameHeuristics allows extra fields without crashing."""
    data = {
        "intro_map_ids": [0],
        "overworld_map_id": 1,
        "auto_pilot_until_map": 0,
        "stagnation_threshold_minutes": 5.0,  # Extra field
        "exploration_entropy": 0.2           # Extra field
    }
    heuristics = GameHeuristics(**data)
    assert heuristics.intro_map_ids == [0]
    # Verify extra field is allowed
    assert "stagnation_threshold_minutes" in heuristics.model_dump()

def test_game_config_extra_fields():
    """Verify that GameConfig allows extra fields at the top level."""
    data = {
        "name": "Test Game",
        "emulator": "gb",
        "rom": "test.gb",
        "controller": "gb",
        "rewards": [],
        "extra_top_level_field": "some_value"
    }
    config = GameConfig(**data)
    assert config.name == "Test Game"
    # Verify extra field is allowed
    assert "extra_top_level_field" in config.model_dump()

def test_game_profile_extra_fields():
    """Verify that GameProfile allows extra fields."""
    data = {
        "intro_guidance": "test guidance",
        "unexpected_profile_field": 123
    }
    profile = GameProfile(**data)
    assert profile.intro_guidance == "test guidance"
    assert "unexpected_profile_field" in profile.model_dump()

def test_ram_layout_extra_fields():
    """Verify that RAMLayout allows extra fields (crucial for custom RAM probes)."""
    data = {
        "map_id": 0xD35E,
        "custom_ram_address": 0xFFFF
    }
    layout = RAMLayout(**data)
    assert layout.map_id == 0xD35E
    assert "custom_ram_address" in layout.model_dump()

def test_checkpoint_extra_fields():
    """Verify that Checkpoint and ConditionConfig allow extra fields."""
    data = {
        "name": "TEST_CHECKPOINT",
        "description": "testing...",
        "condition": {
            "type": "ram",
            "params": {"address": 0x1234, "target_value": 1},
            "extra_condition_field": True
        },
        "extra_checkpoint_field": "hello"
    }
    checkpoint = Checkpoint(**data)
    assert checkpoint.name == "TEST_CHECKPOINT"
    assert "extra_checkpoint_field" in checkpoint.model_dump()
    assert "extra_condition_field" in checkpoint.condition.model_dump()

def test_reward_config_extra_fields():
    """Verify that RewardConfig allows extra fields."""
    data = {
        "type": "exploration",
        "params": {"threshold": 0.1},
        "unknown_reward_setting": 42
    }
    reward = RewardConfig(**data)
    assert reward.type == "exploration"
    assert "unknown_reward_setting" in reward.model_dump()

def test_game_state_extra_fields():
    """Verify that GameState allows extra fields."""
    from autogameplayer.core.models import GameState
    data = {
        "image_data": "abc",
        "vision_vector": [0.1, 0.2],
        "unexpected_state_field": "oops"
    }
    state = GameState(**data)
    assert state.image_data == "abc"
    assert "unexpected_state_field" in state.model_dump()

def test_action_extra_fields():
    """Verify that Action allows extra fields."""
    from autogameplayer.core.models import Action
    data = {
        "button": "a",
        "reasoning": "testing",
        "unexpected_action_field": 123
    }
    action = Action(**data)
    assert action.button == "a"
    assert "unexpected_action_field" in action.model_dump()
