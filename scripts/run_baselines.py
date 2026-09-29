import argparse


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--lightgbm", action="store_true")
    args = parser.parse_args()
    print("Running baseline models")


if __name__ == "__main__":
    main()
