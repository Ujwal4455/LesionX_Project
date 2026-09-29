import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()
    print("Inference image:", args.image)


if __name__ == "__main__":
    main()
