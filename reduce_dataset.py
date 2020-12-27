import random
import argparse
from pathlib import Path
import os
from shutil import copyfile


def get_shuffled_labels(size, train_pct, val_pct):
    labels = ["train"] * (int(size * train_pct))
    labels += ["val"] * (int(size * val_pct))
    labels += ["test"] * (size - len(labels))
    random.shuffle(labels)
    return labels


def reduce_dataset(original_dir, target_dir, size=None, train_pct=0.7,
                   val_pct=0.2, split=True, shuffle=True):

    pathnames = list(Path(original_dir).glob('**/*.jpg'))

    if size is None:
        size = len(pathnames)
    elif size > len(pathnames):
        print("size should <= ", len(pathnames))
        return None

    if shuffle:
        random.shuffle(pathnames)

    if split:
        labels = get_shuffled_labels(size, train_pct, val_pct)

    print("Copying {} files from {} to {}\n\n".format(size, original_dir, target_dir))

    for i in range(size):
        previous_path = str(pathnames[i]).split("\\")[-1]
        subdir = ""

        if split:
            subdir = labels[i]

        new_path = os.path.join(target_dir, subdir, previous_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        copyfile(pathnames[i], new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="directory containing the dataset")
    parser.add_argument("--target", type=str, help="directory to copy files to")
    parser.add_argument("--size", type=int, help="number of files to copy")
    parser.add_argument("--split", type=bool, default=True, help="number of files to copy")
    parser.add_argument("--train_pct", type=float, default=True, help="percentage size of training set")
    parser.add_argument("--val_pct", type=float, default=True, help="percentage size of validation set")
    parser.add_argument("--shuffle", type=bool, default=True, help="randomly choose the files")

    args = parser.parse_args()

    reduce_dataset(original_dir=args.dir, target_dir=args.target,
                   split=args.split, size=args.size, shuffle=args.shuffle)