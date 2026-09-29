from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("Running LesionX configuration check")
    print(cfg)
