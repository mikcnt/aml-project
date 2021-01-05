import argparse


def main_parser():
    # Parse arguments and prepare program
    parser = argparse.ArgumentParser(description='Training and Using ColorizationNet')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to .pth file checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='use this flag to validate without training')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='batch size (default: 12)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of epochs (default: 100)')
    parser.add_argument('--learning_rate', default=3e-5, type=float, metavar='N', help='learning rate (default 3e-5')
    parser.add_argument('--weight_decay', default=1e-3, type=int, metavar='N', help='learning rate (default 3e-5')
    parser.add_argument('--data_dir', default='data', type=str, metavar='N',
                        help='dataset directory, should contain train/test subdirs')
    parser.add_argument('--use_gpu', default=True, type=bool, metavar='B', help='specify whether to use GPU')
    parser.add_argument('--loss', default='classification', type=str, metavar='string',
                        help='specify target loss function')
    parser.add_argument('--alpha', default=.5, type=float, metavar='string',
                        help='weighting factor in smoothing prior probability')
    return parser
