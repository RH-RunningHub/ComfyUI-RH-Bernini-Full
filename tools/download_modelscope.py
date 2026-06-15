#!/usr/bin/env python3
"""Download bytedance-community/Bernini-Diffusers from ModelScope."""

import argparse

from modelscope.hub.snapshot_download import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download Bernini-Diffusers from ModelScope")
    parser.add_argument("--target", required=True, help="Local target directory")
    parser.add_argument("--model", default="bytedance-community/Bernini-Diffusers")
    args = parser.parse_args()
    snapshot_download(args.model, local_dir=args.target)
    print(args.target)


if __name__ == "__main__":
    main()
