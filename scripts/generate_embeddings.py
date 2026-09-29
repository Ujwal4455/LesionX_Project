import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate lesion embeddings")
    parser.add_argument("--output", type=str, default="data/embeddings")
    args = parser.parse_args()
    print(f"Generating embeddings to: {args.output}")


if __name__ == "__main__":
    main()
