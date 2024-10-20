import sys

from diffusion import inference

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    inference(checkpoint_path)