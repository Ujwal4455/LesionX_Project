from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("Training patient model")
    print(cfg)
