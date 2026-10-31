
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("AeroSentinel training pipeline initialized.")
    print("Next: configure target, features, and model.")
    print("No model has been trained yet.")


if __name__ == "__main__":
    main()
