import torch
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import natsort
import PIL.Image as Image



class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, class_names=None):
        self.main_dir = main_dir
        self.transform = transform
        all_classes = os.listdir(main_dir)
        all_imgs = []
        for c in all_classes:
            class_img_names = os.listdir(os.path.join(main_dir, c))
            for img_name in class_img_names:
                all_imgs.extend([os.path.join(main_dir, c, img_name)])
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def show_img(imgs, fig_sz=None, title=None, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    """
        Takes imgs as a torch tensor, clips them to be in [0, 1], shows the imgs.
    """
    if fig_sz:
        plt.figure(figsize=fig_sz)
    # unnormalize
    imgs = (imgs * torch.tensor(std).view(3, 1, 1)) + torch.tensor(mean).view(3, 1, 1)
    npimg = imgs.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(train_data, val_data, model, criterion, optimizer, scheduler, num_epochs=10,
                summary=None, samples=None, device='cpu', dtype=torch.float32, use_ten_crops=False):

    since = time.time()

    model = model.to(device=device)
    data = {'train': train_data, 'val': val_data}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    actual_size = dict()
    train_loss, train_acc = 0.0, 0.0

    if summary is not None:
        writer = SummaryWriter(summary)
        if samples is not None:
            writer.add_graph(model, samples)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start_time = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            actual_size[phase] = 0
            running_loss = 0.0
            running_corrects = 0
            # torch.cuda.empty_cache()

            # Iterate over data.
            for t, (inputs, labels) in enumerate(data[phase]):
                inputs = inputs.to(device=device, dtype=dtype)
                labels = labels.to(device=device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if use_ten_crops and phase == 'val':
                        bs, ncrops, c, h, w = inputs.size()
                        outputs = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
                        outputs = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
                    else:
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                actual_size[phase] += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # del inputs
                # del labels
                # del outputs
                # del preds
                # del loss
                # torch.cuda.empty_cache()

            epoch_loss = running_loss / actual_size[phase]
            epoch_acc = running_corrects.double() / actual_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                scheduler.step(epoch_acc)

            if summary is not None and phase == 'val':
                writer.add_scalars('Loss', {'train': train_loss, 'val': epoch_loss}, epoch)
                writer.add_scalars('Accuracy', {'train': train_acc, 'val': epoch_acc}, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("elapsed_time = {:.0f}s".format(time.time() - epoch_start_time))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(val_data, model, n_batches=1, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                   use_ten_crops=False, device='cpu'):
    model = model.to(device)
    for i, (val_imgs, val_labels) in enumerate(val_data):
        if type(val_imgs) == tuple:
            val_imgs = torch.cat(val_imgs,0)
        val_imgs = val_imgs.to(device, dtype=torch.float32)
        if use_ten_crops:
            bs, ncrops, c, h, w = val_imgs.size()
            outputs = model(val_imgs.view(-1, c, h, w))  # fuse batch size and ncrops
            scores = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        else:
            scores = model(val_imgs)
        _, pred = torch.max(scores, 1)
        if i == 0:
            all_imgs, all_labels, all_preds = val_imgs, val_labels, pred
        else:
            all_imgs = torch.cat((all_imgs, val_imgs), 0)
            all_labels = torch.cat((all_labels, val_labels), 0)
            all_preds = torch.cat((all_preds, pred), 0)
        if i+1 == n_batches:
            break

    if use_ten_crops:
        all_imgs = all_imgs[:, 0].view(-1, c, h, w)
    all_imgs = (all_imgs * torch.tensor(std, device=device).view(3, 1, 1)) +\
               torch.tensor(mean, device=device).view(3, 1, 1)

    all_imgs = all_imgs.cpu().numpy().transpose((0, 2, 3, 1)).clip(0., 1.)
    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    return all_imgs, all_preds, all_labels
