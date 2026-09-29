import argparse


def main():
    parser = argparse.ArgumentParser(description="Train the lesion model")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(f"Training lesion model for {args.epochs} epoch(s)")


if __name__ == "__main__":
    main()
