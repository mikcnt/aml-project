import time
from utils import *


def validate(val_loader, model, criterion, save_images, epoch, use_gpu):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, input_smooth) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu: input_gray, input_ab, input_smooth = input_gray.cuda(), input_ab.cuda(), input_smooth.cuda()

        # Run model and record loss
        output_ab = model(input_gray)
        # loss = criterion(output_ab, input_ab)
        loss = multicrossentropy_loss(output_ab, input_smooth)
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):  # save at most 10 images
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                gray_smooth_tensor2rgb(input_gray[j].cpu(),
                                       img_smooth=output_ab[j].detach().cpu(),
                                       save_path=save_path,
                                       save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                  loss=losses))

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

        # loss = criterion(output_ab, input_ab)
        loss = multicrossentropy_loss(output_ab, input_smooth)
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
