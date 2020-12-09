# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from model import ColorizationNet
from torchvision import datasets, transforms
from dataset_class import GrayscaleImageFolder
from fit import train, validate
# For utilities
import os, shutil, time, argparse
from utils import AverageMeter, to_rgb

# Parse arguments and prepare program
parser = argparse.ArgumentParser(description='Training and Using ColorizationNet')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to .pth file checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')

def main():
    global args, best_losses, use_gpu
    args = parser.parse_args()
    print('Arguments: {}'.format(args))

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()

    # Model definition
    model = ColorizationNet()

    # Loss and optimizer definition
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Resume training if checkpoint path is given
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)# if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
            # args.start_epoch = checkpoint['epoch']
            # best_losses = checkpoint['best_losses']
            model.load_state_dict(checkpoint)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading checkpoint.')# Resuming from epoch {}'.format(checkpoint['epoch']))
        else:
            print('Checkpoint filepath incorrect.')
            return

    # Training data
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder('data/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=16, shuffle=True)

    # Validation data
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    val_imagefolder = GrayscaleImageFolder('data/val', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=16, shuffle=False)
            
    # Move model and loss function to GPU
    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()
        
    # Make folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10
    epochs = 100

    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch, use_gpu)
        with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch, use_gpu)
        # Save checkpoint and replace old best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
            
            
if __name__ == '__main__':
    main()