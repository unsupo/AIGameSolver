import torch
from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.muzero.trainer import MuZeroOfflineTrainer


def test_muzero_model_initial_inference():
    # Setup standard model
    model = MuZeroModel(
        input_dim=10,
        hidden_dim=32,
        action_dim=4,
        map_id_dim=10,
        ocr_hash_dim=10,
        num_personalities=8,
    )

    # Dummy vision vector
    obs = torch.randn(2, 10)

    # Initial inference with UVFA conditioning on personality 0
    h, p, ve, vi = model.initial_inference(obs, personality_id=[0, 1])

    assert h.shape == (2, 32)
    assert p.shape == (2, 4)
    assert ve.shape == (2, 1)  # Extrinsic split head
    assert vi.shape == (2, 1)  # Intrinsic split head


def test_muzero_model_recurrent_inference():
    model = MuZeroModel(
        input_dim=10,
        hidden_dim=32,
        action_dim=4,
        map_id_dim=10,
        ocr_hash_dim=10,
        num_personalities=8,
    )
    h = torch.randn(2, 32)
    actions = torch.tensor([1, 2])

    next_h, reward, map_l, ocr_l, p, ve, vi = model.recurrent_inference(
        h, actions, personality_id=[0, 1]
    )

    assert next_h.shape == (2, 32)
    assert reward.shape == (2, 1)
    assert p.shape == (2, 4)
    assert ve.shape == (2, 1)
    assert vi.shape == (2, 1)


def test_trainer_target_network_initialization():
    model = MuZeroModel(input_dim=10, hidden_dim=32, action_dim=4)
    trainer = MuZeroOfflineTrainer(model, lr=1e-4)

    # Target should be distinct from model
    assert id(trainer.model) != id(trainer.target_model)

    # Weights should be identical initially
    for p1, p2 in zip(trainer.model.parameters(), trainer.target_model.parameters()):
        assert torch.allclose(p1, p2)

    # After a fake gradient step, they should diverge until sync
    trainer.optimizer.zero_grad()
    dummy_loss = trainer.model.prediction.value_ext.weight.sum()
    dummy_loss.backward()
    trainer.optimizer.step()

    match_count = 0
    for p1, p2 in zip(trainer.model.parameters(), trainer.target_model.parameters()):
        if torch.allclose(p1, p2):
            match_count += 1

    # Not all parameters should match exactly (some weights updated)
    assert match_count < len(list(trainer.model.parameters()))
