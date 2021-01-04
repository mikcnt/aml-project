import torch.nn as nn
from model import ColorizationNet
from torchvision import transforms
from data_loader import GrayscaleImageFolder
from fit import train, validate
import os
import argparse
from utils import *

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
parser.add_argument('--loss', default='classification', type=str, metavar='string', help='specify target loss function')
parser.add_argument('--alpha', default=.5, type=float, metavar='string',
                    help='weighting factor in smoothing prior probability')


def main():
    args = parser.parse_args()
    print('Arguments: {}'.format(args))

    # Check if GPU is available
    use_gpu = args.use_gpu and torch.cuda.is_available()

    start_epoch = 0

    # Model definition
    model = ColorizationNet(args.loss)

    # Loss and optimizer definition
    criterion = CustomLoss(args.loss, args.alpha)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    train_set_path = os.path.join(args.data_dir, 'train')
    test_set_path = os.path.join(args.data_dir, 'val')

    # Training data
    # train_transforms = transforms.Compose([transforms.RandomRotation(45), transforms.RandomHorizontalFlip(),
    #                                    transforms.Resize((h, w))])
    resize_crop = transforms.RandomApply([transforms.Resize((h + h // 10, w + w // 10)),
                                          transforms.CenterCrop((h, w))], p=0.2)

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((h, w)), resize_crop])

    train_imagefolder = GrayscaleImageFolder(train_set_path, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    # Validation data
    val_transforms = transforms.Compose([transforms.Resize((h, w))])
    val_imagefolder = GrayscaleImageFolder(test_set_path, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder,
                                             batch_size=args.batch_size,
                                             shuffle=False)

    # Make folders and set parameters
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10
    epochs = args.epochs
    train_loss = {}
    validate_loss = {}

    # Resume training if checkpoint path is given
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_losses = checkpoint['best_losses']
            train_loss = checkpoint['train_loss']
            validate_loss = checkpoint['validate_loss']

            print('Finished loading checkpoint.')
            print("Resuming from epoch {}".format(checkpoint['epoch']))
        else:
            print('Checkpoint filepath incorrect.')
            return

    # Move model and loss function to GPU
    if use_gpu:
        model = model.cuda()

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if not args.evaluate:
        # Train model
        for epoch in range(start_epoch + 1, epochs):
            # Train for one epoch, then validate
            losses_avg = train(train_loader, model, criterion, optimizer, epoch, use_gpu)
            train_loss[epoch] = int(losses_avg)

            with torch.no_grad():
                losses = validate(val_loader, model, criterion, save_images, epoch, use_gpu)
                validate_loss[epoch] = int(losses)

            checkpoint_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_losses': best_losses,
                'train_loss': train_loss,
                'validate_loss': validate_loss
            }

            # Save checkpoint every 25 epochs or when a better model is produced
            if losses < best_losses:
                best_losses = losses
                torch.save(checkpoint_dict, 'checkpoints/best-model.pth')

            # save model on each epoch
            if epoch % 5 == 0:
                torch.save(checkpoint_dict,
                           'checkpoints/model-epoch-{}-losses-{:.0f}.pth'.format(epoch, int(losses)))
    else:
        with torch.no_grad():
            validate(val_loader, model, criterion, save_images, 0, use_gpu)


if __name__ == '__main__':
    main()
