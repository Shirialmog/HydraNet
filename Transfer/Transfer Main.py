import torch.backends.cudnn as cudnn
import torch
import numpy as np
import torch.nn as nn
from Transfer.SettingsTransfer import SegSettings
import random
import json
import time
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import logging
import sys
from Transfer.Model_Transfer import Unet_2D as Unet_2D_transfer
import os
import copy

matplotlib.use('TkAgg')
cudnn.benchmark = True

user='remote'
device = 'cuda:0'

class Seg_Dataset(BaseDataset):
    def __init__(self, task,images_dir,masks_dir, num_classes, transforms=None):
        self.task=task
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self,idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])
        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])

        if settings.pre_process==True:
            image = pre_processing(image,self.task,settings)
        if settings.augmentation==True:
            image, mask = create_augmentations(image, mask)

        sample={'image':image.astype('float64'), 'mask':mask.astype('float64'), 'task':self.task, 'num_classes':self.num_classes }
        return sample

    def __len__(self):
        return len(os.listdir(self.images_dir))

class DiceLoss(nn.Module):
    def __init__(self, classes, dimension, mask_labels_numeric, mask_class_weights_dict, is_metric):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.dimension = dimension
        self.mask_labels_numeric = mask_labels_numeric #actul pixel value [0,1,2]
        self.mask_class_weights_dict = mask_class_weights_dict #weight of class
        self.is_metric = is_metric #if metric, return 1-DiceLoss
        self.eps = 1e-10
        self.tot_weight = torch.sum(torch.Tensor(list(mask_class_weights_dict.values()))).item()

    def forward(self, pred, target):
        if self.is_metric:
            if self.classes >1:
                pred = torch.argmax(pred, dim=1)
                pred = torch.eye(self.classes)[pred]
                pred = pred.transpose(1, 3).to(device)
            else:
                pred_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                pred_copy[:, 1, :, :][pred[:, 0, :, :]  > 0.5] = 1
                pred_copy[:, 0, :, :][pred[:, 0, :, :] <= 0.5] = 1
                target_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                target_copy[:, 1, :, :][target[:, 0, :, :] == 1.0] = 1
                target_copy[:, 0, :, :][target[:, 0, :, :] == 0.0] = 1

                pred = pred_copy
                target = target_copy.to(device)
        target=target.transpose(2,3)
        batch_intersection = torch.sum(pred * target.float(), dim=tuple(list(range(2, self.dimension + 2))))
        batch_union = torch.sum(pred, dim=tuple(list(range(2, self.dimension + 2)))) + torch.sum(target.float(),dim=tuple(list(range(2,self.dimension + 2))))
        background_dice = (2 * batch_intersection[:, self.mask_labels_numeric['background']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['background']] + self.eps)
        organ_dice = (2 * batch_intersection[:, self.mask_labels_numeric['organ']] + self.eps) / (
                batch_union[:, self.mask_labels_numeric['organ']] + self.eps)

        mean_dice_val = torch.mean((background_dice * self.mask_class_weights_dict['background'] +
                                    organ_dice * self.mask_class_weights_dict['organ']) * 1 / self.tot_weight, dim=0)

        if 'tumour' in self.mask_labels_numeric:
            tumour_dice = (2 * batch_intersection[:, self.mask_labels_numeric['tumour']] + self.eps) / (
                    batch_union[:, self.mask_labels_numeric['tumour']] + self.eps)
            mean_dice_val = torch.mean((background_dice * self.mask_class_weights_dict['background'] +
                                        organ_dice * self.mask_class_weights_dict['organ']+tumour_dice * self.mask_class_weights_dict['tumour']) * 1 / self.tot_weight,dim=0)
            if self.is_metric:
                return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),tumour_dice.mean().item()]

        if self.is_metric:
            return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item()]
        else:
            return -mean_dice_val

def dice(pred, target, num_classes,settings):
    """
    This function receives a prediction and target of same size,
    and returns the dice score calculated
    output: [mean_dice, organ_dice, background dice]
    """
    if num_classes==2:
        mask_labels={'background': 0,  'organ': 1}
        loss_weights= {'background': 1, 'organ': 10}
    elif num_classes==3: ## pancreas only
        mask_labels = {'background': 0, 'organ': 1, 'tumour':2}
        loss_weights = {'background': 1, 'organ': 10,'tumour':20}
    dice_measurement = DiceLoss(classes=num_classes,
                               dimension=settings.dimension,
                               mask_labels_numeric=mask_labels,
                               mask_class_weights_dict=loss_weights,
                               is_metric=True)
    [*dices] = dice_measurement(pred, target)
    return dices

def create_augmentations(image,mask):
    """
    This function randomly chooses slices on which to perform augmentation
    augmentation types:
    1. 90 deg rotation CW
    2. flip upside down
    3. flip left to right
    4. 90 deg rotation CCW
    """
    augmentation=np.rot90
    p=random.choice([0,1,2])  #33.33 % chance to perform augmentation
    if p==1:
        new_image=np.zeros(image.shape)
        new_mask=np.zeros(mask.shape)
        k=random.choice([0,1,2,3])
        if k==0:
            augmentation = np.rot90
        if k==1:
            augmentation = np.fliplr
        if k==2:
            augmentation = np.flipud
        if k==3:
            augmentation = np.rot90
            new_mask = augmentation(mask,axes=(1,0))
            for i in range(image.shape[0]):
                new_image[i,:,:] = augmentation(image[i,:,:],axes = (1,0))
            return (new_image.copy(), new_mask.copy())

        new_mask = augmentation(mask)
        for i in range(image.shape[0]):
            new_image[i,:,:] = augmentation(image[i,:,:])
        return (new_image.copy(), new_mask.copy())

    else: #no agumentation
        return(image,mask)

def pre_processing(input_image, task, settings):
    if task == ('spleen' or 'lits' or 'pancreas' or 'hepatic vessel'):  # CT, clipping, Z_Score, normalization btw0-1
        clipped_image = clip(input_image, settings)
        c_n_image = zscore_normalize(clipped_image)
        min_val = np.amin(c_n_image)
        max_val = np.amax(c_n_image)
        eps=0.000001
        final = (c_n_image - min_val) / (max_val - min_val+eps)
        final[final > 1] = 1
        final[final < 0] = 0

    else: #MRI, Z_score, normalization btw 0-1
        norm_image = zscore_normalize(input_image)
        min_val = np.amin(norm_image)
        max_val = np.amax(norm_image)
        eps = 0.000001
        final = (norm_image - min_val) / (max_val - min_val+eps)
        final[final > 1] = 1
        final[final < 0] = 0
        final=norm_image

    return final

def clip(data, settings):
    """
    Set min and max values by windowing
    """
    min_val = settings.min_clip_val
    max_val = settings.max_clip_val
    data[data > max_val] = max_val
    data[data < min_val] = min_val

    return data

def clip_n_normalize(data, settings):
    # clip and normalize
    min_val = settings.min_clip_val
    max_val = settings.max_clip_val
    data = ((data - min_val) / (max_val - min_val))
    data[data > 1] = 1
    data[data < 0] = 0

    return data

def zscore_normalize(img):
    eps=0.0001
    mean = img.mean()
    std = img.std()+eps
    normalized = (img - mean) / std
    return normalized

def save_samp(image,mask,task,prediction,epoch,iter,snapshot_dir,loss):
    """
    save prediction and mask for visualization
    """
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction[0]
    prediction = np.squeeze(prediction)

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Dice: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def save_samp_validation(image,mask,task,prediction,epoch,iter,loss,settings):
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = prediction[0]
    prediction = np.squeeze(prediction)
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Dice: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(settings.validation_snapshot_dir,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
    plt.close('all')

def my_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        filename=logger_name,
        filemode='w',
        format='%(asctime)s, %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    logger.addHandler(file_handler)
    return logger

def plot_graph(num_epochs,settings,data_dic):
    loss_type = ['CE', 'bg_dice', 'organ_dice']
    set_type = ['Training', 'Validation']
    colors = ['r', 'b', 'c', 'y', 'k', 'g']

    plt.figure()
    x = np.arange(0, num_epochs, 1)
    i = 0
    for s in set_type:
        for l in loss_type:
            plt.plot(x, data_dic[s][l]['total_epochs'], colors[i])
            i += 1
    plt.title('Training & Validation loss and dice vs num of epochs')
    plt.legend(
        ['Train Organ Dice', 'Train Background Dice', 'Train CE Loss', 'Val Organ Dice', 'Val Background Dice',
         'Val CE Loss'], loc='upper left')
    plt.savefig(os.path.join(settings.snapshot_dir,
                             'Training & Validation loss and dice vs num of epochs.png'))
    return


def dataloader_transfer( data_folder,task,batch_size):
    train_dataset = Seg_Dataset(task, os.path.join(data_folder,'Training'),os.path.join(data_folder,'Training_Labels'),2)
    val_dataset = Seg_Dataset(task, os.path.join(data_folder,'Validation'),os.path.join(data_folder,'Validation_Labels'),2)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader,val_dataloader

def train_transfer(setting_dict, task,data_folder, exp_ind, model):
    logger = my_logger(settings.simulation_folder+'\logger')
    model.to(device)
    model = model.double()
    criterion = nn.CrossEntropyLoss() #smp.utils.losses.DiceLoss() # smp_DiceLoss()#
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    # Initialize 'data_dic', nested dictionary, will contain all losses and dice
    count_type = ['cur', 'total_epochs']
    loss_type = ['CE', 'bg_dice', 'organ_dice']
    set_type = ['Training', 'Validation']

    partition_by_count = dict(zip(count_type, [list() for i in count_type]))
    partition_by_dice = dict(zip(loss_type, [copy.deepcopy(partition_by_count) for i in loss_type]))
    data_dic = dict(zip(set_type, [copy.deepcopy(partition_by_dice) for i in set_type]))  ##this is the final dic

    batch_size = setting_dict.batch_size
    train_dataloader, val_dataloader=dataloader_transfer(data_folder,task, batch_size)

    print('Training... ')
    num_epochs=2

    for epoch in range(0, num_epochs):
         epoch_start_time = time.time()
         total_steps = len(train_dataloader)
         for i,sample in enumerate(train_dataloader,1):
             model.train()
             print(sample['task'])
             images=sample['image'].double()
             masks = sample['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             masks = masks.type(torch.LongTensor)
             images=images.to(device)
             masks = masks.to(device)

             #Forward pass
             outputs = model(images,sample['task'])
             outputs = outputs.to(device)
             loss = criterion(outputs.double(), masks[:,0,:,:])

             # Backward and optimize
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             logger.info('current task: ' + sample['task'][0])
             logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )


             one_hot = torch.DoubleTensor(masks.size(0), sample['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             activation = nn.Softmax(dim=1)
             dices = dice(activation(outputs), masks_dice, sample['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]

             data_dic['Training']['organ_dice']['cur'].append(organ_dice)
             data_dic['Training']['bg_dice']['cur'].append(background_dice)
             data_dic['Training']['CE']['cur'].append(loss.item())
             if i % 30 == 0:
                 save_samp(sample['image'][0], sample['mask'][0], sample['task'][0], outputs, epoch, i,
                           settings.snapshot_dir, organ_dice)

             if (i + 1) % 50 == 0:
                 print('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                       'iter: {}/{}'.format(np.mean(data_dic['Training']['CE']['cur']),
                                                   np.mean(data_dic['Training']['organ_dice']['cur']),
                                                   np.mean(data_dic['Training']['bg_dice']['cur']),
                                            i + 1, len(train_dataloader)))
                 logger.info('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                              'iter: {}/{}'.format(np.mean(data_dic['Training']['CE']['cur']),
                                                   np.mean(data_dic['Training']['organ_dice']['cur']),
                                                   np.mean(data_dic['Training']['bg_dice']['cur']),
                                                   i + 1, len(train_dataloader)))

         for l in loss_type:
            data_dic['Training'][l]['total_epochs'].append(np.mean(data_dic['Training'][l]['cur']))

         print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,
                                                                   (time.time() - epoch_start_time) / 60))
         torch.save(model.state_dict(), os.path.join(settings.model_dir, 'exp_{}_epoch_{}.pt'.format(exp_ind,epoch)))
         torch.save(model.encoder.state_dict(), os.path.join(settings.model_dir, 'encoder_exp_{}_epoch_{}.pt'.format(exp_ind,epoch)))


         total_steps=len(val_dataloader)
         for i, data in enumerate(val_dataloader):
             torch.no_grad()

             images = data['image'].double()
             masks = data['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             images = images.to(device)
             masks = masks.to(device)

             outputs = model(images, data['task'])
             outputs = outputs.to(device)

             loss = criterion(outputs.double(), masks[:,0,:,:])
             logger.info('current task: ' + sample['task'][0])
             logger.info(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

             one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2), masks.size(3)).zero_()
             one_hot = one_hot.to(device)
             masks_dice = one_hot.scatter_(1, masks.data, 1)
             activation = nn.Softmax(dim=1)
             dices = dice(activation(outputs), masks_dice, data['num_classes'][0], settings)
             mean_dice = dices[0]
             background_dice = dices[1]
             organ_dice = dices[2]

             if i%10==0:
                 save_samp_validation(data['image'][0], data['mask'][0], data['task'][0], outputs, epoch, i,
                                      organ_dice,settings)

             data_dic['Validation']['organ_dice']['cur'].append(organ_dice)
             data_dic['Validation']['bg_dice']['cur'].append(background_dice)
             data_dic['Validation']['CE']['cur'].append(loss.item())

         for l in loss_type:
            data_dic['Validation'][l]['total_epochs'].append(np.mean(data_dic['Training'][l]['cur']))

         print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,(time.time() - epoch_start_time) / 60))
         print('train loss: {} val_loss: {}'.format(np.mean(data_dic['Training']['CE']['total_epochs']),
                                                    np.mean(data_dic['Validation']['CE']['total_epochs'])))

         np.save(settings.dice_dir + r'\train_organ_epoch_{}.npy'.format(epoch),
                 np.array(data_dic['Training']['organ_dice']['total_epochs']))
         np.save(settings.dice_dir + r'\val_organ_epoch_{}.npy'.format(epoch),
                 np.array(data_dic['Validation']['organ_dice']['total_epochs']))

    plot_graph(num_epochs,settings,data_dic)

if __name__ == '__main__':
    exp_ind = 3 #experiment to start with
    json_path = r'G:\Deep learning\Datasets_organized\small_dataset\Transfer_exp\Uterus\truncate_0.2\100\exp_3\imagenet\exp_3.json'
    with open(json_path) as json_file:
        settings = json.load(json_file)

    settings = SegSettings(settings, write_logger=True)

    model = Unet_2D_transfer(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)
    if settings.exp_type=='transfer':
        encoder_path = r'G:\Deep learning\Datasets_organized\small_dataset\Experiments\exp_3_imagenet_final\checkpoint\encoder_exp_3_epoch_14.pt'
        model.get_encoder('densenet121',weights=encoder_path)
        print('model loaded')
    percentage=settings.percentage
    task = 'Uterus'
    data_folder = r'G:\Deep learning\Datasets_organized\Prepared_Data\Uterus_truncate_0.2\100'
    train_transfer(settings, task, data_folder, exp_ind, model)
