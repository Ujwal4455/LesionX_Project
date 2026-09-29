import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect SLICE-3D dataset")
    parser.add_argument("--root", default=None, help="Dataset root path")
    args = parser.parse_args()

    root = args.root or os.getenv("LESIONX_DATASET_ROOT")
    if not root:
        print("No dataset root provided. Set dataset.root in config.yaml or use --root.")
        return

    from src.data.schema import inspect_dataset
    inspect_dataset(root)


if __name__ == "__main__":
    main()
