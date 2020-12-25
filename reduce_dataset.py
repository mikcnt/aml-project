import random
import argparse
from pathlib import Path
import os
from shutil import copyfile

import args as args


def reduce_dataset(original_dir, target_dir, size=None, shuffle=True):
    assert (size is not None)

    pathnames = list(Path(original_dir).glob('**/*.jpg'))

    if size > len(pathnames):
        print("size should <= ", len(pathnames))
        return None

    if shuffle:
        random.shuffle(pathnames)

    for i in range(size):
        previous_path = str(pathnames[i]).replace(original_dir, '')
        new_path = os.path.join(target_dir, previous_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        copyfile(pathnames[i], new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="directory containing the dataset")
    parser.add_argument("--target", type=str, help="directory to copy files to")
    parser.add_argument("--size", type=int, help="number of files to copy")
    parser.add_argument("--shuffle", type=bool, default=True, help="randomly choose the files")

    args = parser.parse_args()

    reduce_dataset(args.dir, args.dir, size=args.size, shuffle=args.shuffle)