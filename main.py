import torch.nn as nn
from model import ColorizationNet
from torchvision import transforms
from data_loader import GrayscaleImageFolder
from fit import train, validate
import os
from config import main_parser
from utils import *


def main():
    parser = main_parser()
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
            losses = train(train_loader, model, criterion, optimizer, epoch, use_gpu)
            train_loss[epoch] = int(losses)

            with torch.no_grad():
                losses = validate(val_loader, model, criterion, epoch, use_gpu)
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
            validate(val_loader, model, criterion, 0, use_gpu)


if __name__ == '__main__':
    main()
