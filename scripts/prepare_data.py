from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    dataset_root = cfg.get("dataset", {}).get("root")
    print("Dataset root:", dataset_root or "not configured")
    print("Preparing data for SLICE-3D discovery.")
    print("Set dataset.root in config.yaml before running training.")
