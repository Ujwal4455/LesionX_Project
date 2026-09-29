from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("LesionX training pipeline")
    print(cfg.get("project", {}).get("name", "LesionX Vision AI"))
