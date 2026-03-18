import asyncio
import argparse
import torch
from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.muzero.trainer import MuZeroOfflineTrainer
from autogameplayer.core.config import settings


async def main():
    parser = argparse.ArgumentParser(description="MuZero Offline Trainer")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save model every N epochs"
    )
    args = parser.parse_args()

    print("🚀 Initializing MuZero Offline Training...")

    # Initialize tripartite model
    # Note: Using default dimensions (384 input, 256 hidden, 8 actions)
    model = MuZeroModel(input_dim=384, hidden_dim=256, action_dim=8)

    # Load existing weights if they exist
    weights_path = settings.models_dir / "muzero_weights.pth"
    if weights_path.exists():
        print(f"🧠 Loading existing weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    trainer = MuZeroOfflineTrainer(model, lr=args.lr)

    print(f"📊 Training on device: {trainer.device}")

    for epoch in range(1, args.epochs + 1):
        loss = await trainer.train_step(batch_size=args.batch_size)

        if epoch % 10 == 0 or epoch == 1:
            print(f"📅 Epoch {epoch}/{args.epochs} | Loss: {loss:.6f}")

        if epoch % args.save_every == 0:
            trainer.save_model()

    trainer.save_model()
    print("🏁 Offline training complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Training interrupted by user.")
