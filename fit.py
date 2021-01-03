import pickle
import time
from torchvision import transforms
import torchvision
import os
from utils import *


def validate(val_loader, model, criterion, save_images, epoch, use_gpu):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    images = []
    # already_saved_images = False
    for i, (input_gray, input_ab, input_smooth) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu:
            input_gray, input_ab, input_smooth = input_gray.cuda(), input_ab.cuda(), input_smooth.cuda()

        # Run model and record loss
        output_ab = model(input_gray)

        if criterion.type == 'classification':
            loss = criterion(output_ab, input_smooth)
            image2rgb = gray_smooth_tensor2rgb  # image to save epoch grids
        elif criterion.type == 'regression':
            if use_gpu:
                output_ab = output_ab.cuda()

            loss = criterion(output_ab, input_ab)
            image2rgb = gray_ab_tensor2rgb  # image to save epoch grids

        losses.update(loss.item(), input_gray.size(0))

        # Create img grid
        if len(images) <= 64:
            for j in range(len(output_ab)):
                if len(images) >= 64:
                    break
                images.append(image2rgb(input_gray[j].cpu(),
                                        output_ab[j].detach().cpu()))

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                  loss=losses))

    # Save img grid to file
    images = [transforms.ToTensor()(img) for img in images]
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    path_folder = 'outputs'
    path_file = 'output-epoch-{}.jpg'.format(epoch)
    path = os.path.join(path_folder, path_file)

    torchvision.utils.save_image(grid_img, path)

    print('Finished validation.')
    return losses.avg


def train(train_loader, model, criterion, optimizer, epoch, use_gpu):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()

    for i, (input_gray, input_ab, input_smooth) in enumerate(train_loader):

        # Use GPU if available
        if use_gpu:
            input_gray, input_ab, input_smooth = input_gray.cuda(), input_ab.cuda(), input_smooth.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray)

        if criterion.type == 'classification':
            loss = criterion(output_ab, input_smooth)
        elif criterion.type == 'regression':
            if use_gpu:
                output_ab = output_ab.cuda()

            loss = criterion(output_ab, input_ab)

        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                  i,
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time,
                                                                  loss=losses))

    print('Finished training epoch {}'.format(epoch))

    return losses.avg
