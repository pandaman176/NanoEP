"""Compile and load the CUDA extension: python -m nano_ep.build."""

from .backends._build import load_extension


def main():
    print(load_extension().__file__)


if __name__ == "__main__":
    main()
