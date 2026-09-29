from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("Temporal model training is disabled unless a verified visit/lesion mapping exists.")
    print(cfg)
