import argparse


def main():
    parser = argparse.ArgumentParser(description="Train temporal model")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(f"Training temporal model for {args.epochs} epoch(s)")


if __name__ == "__main__":
    main()
