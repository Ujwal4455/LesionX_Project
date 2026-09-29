import argparse


def main():
    parser = argparse.ArgumentParser(description="Train patient model")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(f"Training patient model for {args.epochs} epoch(s)")


if __name__ == "__main__":
    main()
