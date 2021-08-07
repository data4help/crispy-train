import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="VAE Package")

    parser.add_argument(
        "-c",
        "--config",
        metavar="STRING",
        type=str,
        required=True,
        help="Path to config.",
    )
    parser.add_argument(
        "-t",
        "--task",
        metavar="STRING",
        type=str,
        required=True,
        help="Indication whether using images or music",
    )

    args = parser.parse_args()
    return args
