from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("Generating lesion embeddings")
    print(cfg.get("paths", {}).get("embeddings", "data/embeddings"))
