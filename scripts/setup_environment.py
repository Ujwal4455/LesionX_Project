import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="LesionX environment setup")
    parser.add_argument("--root", default=None, help="Dataset root")
    args = parser.parse_args()
    print("LesionX environment setup complete.")
    print("Detected root:", args.root or os.getenv("LESIONX_DATASET_ROOT") or "not configured")


if __name__ == "__main__":
    main()
