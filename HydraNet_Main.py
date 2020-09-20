import os
import numpy as np
import json
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import logging
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Subset
from torchvision import transforms
from SegmentationModule.SegmentationSettings import SegSettings
import Models as models


matplotlib.use('TkAgg')
cudnn.benchmark = True


class Seg_Dataset(BaseDataset):
    """
    Create a dataset that will be given to the dataloader
    functions: getitem, len
    """
    def __init__(self, task,images_dir,masks_dir, num_classes,settings, transforms=None):
        self.task=task
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.num_classes = num_classes
        self.settings=settings
        self.device = "cuda"

    def __getitem__(self,idx):
        images = os.listdir(self.images_dir)
        image = np.load(self.images_dir + '/' + images[idx])
        masks = os.listdir(self.masks_dir)
        mask = np.load(self.masks_dir + '/' + masks[idx])

        #based on Json settings, perform pre-processing and augmentation
        if self.settings.pre_process==True:
            image = pre_processing(image,self.task,self.settings)
        if self.settings.augmentation==True:
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
        self.is_metric = is_metric  #if metric, return 1-DiceLoss
        self.eps = 1e-10
        self.tot_weight = torch.sum(torch.Tensor(list(mask_class_weights_dict.values()))).item()

    def forward(self, pred, target):
        if self.is_metric:
            if self.classes >1:
                pred = torch.argmax(pred, dim=1)
                pred = torch.eye(self.classes)[pred]
                pred = pred.transpose(1, 3).cuda(1)
            else:
                pred_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                pred_copy[:, 1, :, :][pred[:, 0, :, :]  > 0.5] = 1
                pred_copy[:, 0, :, :][pred[:, 0, :, :] <= 0.5] = 1
                target_copy = torch.zeros((pred.size(0), 2, pred.size(2), pred.size(3)))
                target_copy[:, 1, :, :][target[:, 0, :, :] == 1.0] = 1
                target_copy[:, 0, :, :][target[:, 0, :, :] == 0.0] = 1
                pred = pred_copy
                target = target_copy.cuda(1)

        batch_intersection = torch.sum(pred * target.float(), dim=tuple(list(range(2, self.dimension + 2))))
        batch_union = torch.sum(pred, dim=tuple(list(range(2, self.dimension + 2)))) + torch.sum(target.float(),
                            dim=tuple(list(range(2, self.dimension + 2))))
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
                                        organ_dice * self.mask_class_weights_dict['organ']+tumour_dice * self.mask_class_weights_dict['tumour']) * 1 / self.tot_weight,
                                       dim=0)
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
    if num_classes==2: #background and organ
        mask_labels={'background': 0,  'organ': 1}
        loss_weights= {'background': 1, 'organ': 10}
    elif num_classes==3: ## pancreas only - background, organ and tumour
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
    p=random.choice([0,1,2])  #33.33 % chance to perform augmentation
    if p==1:
        new_image=np.zeros(image.shape)
        k=random.choice([0,1,2,3]) #4 types of augmentation
        if k==0:
            augmentation = np.rot90
        elif k==1:
            augmentation = np.fliplr
        elif k==2:
            augmentation = np.flipud
        elif k==3:
            augmentation = np.rot90
            new_mask = augmentation(mask,axes=(1,0))#axes=(1,0) = CCW
            for i in range(image.shape[0]):
                new_image[i,:,:] = augmentation(image[i,:,:],axes = (1,0))
            return (new_image.copy(), new_mask.copy())

        new_mask = augmentation(mask)
        for i in range(image.shape[0]):
            new_image[i,:,:] = augmentation(image[i,:,:])
        return (new_image.copy(), new_mask.copy())
    else:
        return(image,mask)

def pre_processing(input_image, task, settings):
    if task == ('spleen' or 'lits' or 'pancreas' or 'hepatic vessel'): #CT images: clipping, Z_Score, normalization btw 0-1
        clipped_image = clip(input_image, settings)
        c_n_image = zscore_normalize(clipped_image)
        min_val = np.amin(c_n_image)
        max_val = np.amax(c_n_image)
        eps=0.000001
        final = (c_n_image - min_val) / (max_val - min_val+eps)  #normalize btw 0-1
        final[final > 1] = 1
        final[final < 0] = 0

    else: #MRI images: Z_score, normalization btw 0-1
        norm_image = zscore_normalize(input_image)
        min_val = np.amin(norm_image)
        max_val = np.amax(norm_image)
        eps = 0.000001
        final = (norm_image - min_val) / (max_val - min_val+eps) #normalize btw 0-1
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
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+ str(epoch) + ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(snapshot_dir,task,'pred_ep_{}_it_{}.{}'.format(epoch,iter, 'png')))
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

    organs = ['all_organs', 'spleen', 'prostate', 'lits', 'brain', 'pancreas', 'hepatic_vessel', 'left_atrial']
    loss_type = ['CE', 'bg_dice', 'organ_dice']
    set_type = ['Training', 'Validation']
    colors=['r','b','c','y','k','g']

    for organ in organs:
        plt.figure()
        x = np.arange(0, num_epochs, 1)
        i=0
        for s in set_type:
            for l in loss_type:
                plt.plot(x, data_dic[s][l]['total_epochs'][organ], colors[i])
                i+=1
        plt.title('Training & Validation loss and dice vs num of epochs - {}'.format(organ))
        plt.savefig(os.path.join(settings.snapshot_dir, 'Training & Validation loss and dice vs num of epochs - {}.png'.format(organ)))
    return

def train(setting_dict):
    settings = SegSettings(setting_dict, write_logger=True)
    my_logger(settings.simulation_folder + '\logger')

    # Initialize model:
    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)
    model.cuda(1)
    model = model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.initial_learning_rate)

    # Initialize 'data_dic', nested dictionary, will contain all losses and dice for all organs
    organs = ['all_organs' ,'spleen', 'prostate','lits','brain','pancreas','hepatic_vessel','left_atrial']
    count_type = ['total_epochs', 'cur']
    loss_type = ['CE', 'bg_dice', 'organ_dice']
    set_type=['Training','Validation']

    partition_by_organ = dict(zip(organs, [list() for i in organs])) #first dic - organs
    partition_by_count = dict(zip(count_type, [partition_by_organ for i in count_type]))
    partition_by_dice = dict(zip(loss_type, [partition_by_count.copy() for i in loss_type]))
    data_dic = dict(zip(set_type, [partition_by_dice.copy() for i in set_type])) ##this is the final dic
    ## data dic shape:
    #  {Training: { Cross Entropy: {all epochs: {spleen:[], prostate:[] etc}
    #                              {cur_epoch: {spleen:[], prostate:[] etc}}
    #               organ dice:    {all epochs: {spleen:[], prostate:[] etc}
    #     #                        {cur_epoch: {spleen:[], prostate:[] etc}}
    #               background dice: {all epochs: {spleen:[], prostate:[] etc}
    #     #                          {cur_epoch: {spleen:[], prostate:[] etc}}}}
    #  {Validation: { Cross Entropy: {all epochs: {spleen:[], prostate:[] etc}
    #                              {cur_epoch: {spleen:[], prostate:[] etc}}
    #               organ dice:    {all epochs: {spleen:[], prostate:[] etc}
    #     #                        {cur_epoch: {spleen:[], prostate:[] etc}}
    #               background dice: {all epochs: {spleen:[], prostate:[] etc}
    #     #                          {cur_epoch: {spleen:[], prostate:[] etc}}}}


    #Initialize datasets
    train_dataset_list=[]
    val_dataset_list=[]
    for organ in organs[1:]:

        organ_train_dataset=Seg_Dataset(organ, settings.definition_file_dir +'/'+ organ+'/Training',
                                       settings.definition_file_dir +'/'+ organ + '/Training_Labels', 2,settings)
        organ_val_dataset = Seg_Dataset(organ, settings.definition_file_dir +'/'+ organ+ '/Validation',
                                          settings.definition_file_dir +'/'+ organ + '/Validation_Labels', 2, settings)
        train_dataset_list.append(organ_train_dataset)
        val_dataset_list.append(organ_val_dataset)

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
    print (len(train_dataset))

    batch_size = settings.batch_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Training... ')
    num_epochs=3
    for epoch in range(0, num_epochs):
         epoch_start_time = time.time()
         total_steps = len(train_dataloader)
         for i,sample in enumerate(train_dataloader,1):
             if i>50:
                 break
             model.train()
             images=sample['image'].double()
             masks = sample['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             images=images.to("cuda:1")
             masks = masks.to("cuda:1")
             masks = masks.type(torch.LongTensor)
             masks=masks.cuda(1)


             #Forward pass
             outputs = model(images,sample['task'])
             outputs = outputs.to("cuda:1")
             loss = criterion(outputs.double(), masks[:,0,:,:])

             # Backward and optimize
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             logging.info('current task: ' + sample['task'][0])
             logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

             dices = dice(outputs, masks, sample['num_classes'][0], settings)
             background_dice = dices[1]
             organ_dice = dices[2]

             #Update data dic for relevant organ
             data_dic['Training']['organ_dice']['cur'][sample['task'][0]].append(organ_dice)
             data_dic['Training']['bg_dice']['cur'][sample['task'][0]].append(background_dice)
             data_dic['Training']['CE']['cur'][sample['task'][0]].append(loss.item())

             #update data dic [all organ]
             data_dic['Training']['organ_dice']['cur']['all_organs'].append(organ_dice)
             data_dic['Training']['bg_dice']['cur']['all_organs'].append(background_dice)
             data_dic['Training']['CE']['cur']['all_organs'].append(loss.item())

             if i % 30 == 0:  #save output every 30 batches
                 save_output = outputs.cpu().detach().numpy()
                 save_samp(sample['image'][0], sample['mask'][0], sample['task'][0], save_output[0][1], epoch, i,
                           settings.snapshot_dir, organ_dice)

             if i % 50 == 0: #print details every 50 batches
                 print('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                       'iter: {}/{}'.format(np.mean(data_dic['Training']['CE']['cur']['all_organs']),
                                            data_dic['Training']['organ_dice']['cur']['all_organs'],
                                            np.mean(data_dic['Training']['bg_dice']['cur']['all_organs']),
                                            i + 1, len(train_dataloader)))
                 logging.info('curr train loss: {}  train organ dice: {}  train background dice: {} \t'
                              'iter: {}/{}'.format(np.mean(data_dic['Training']['CE']['cur']['all_organs']),
                                            data_dic['Training']['organ_dice']['cur']['all_organs'],
                                            np.mean(data_dic['Training']['bg_dice']['cur']['all_organs']),
                                            i + 1, len(train_dataloader)))

         #Update data_dic['total_epochs']
         for l in loss_type:
            for organ in organs:
                data_dic['Training'][l]['total_epochs'][organ].append(np.mean(data_dic['Training'][l]['cur'][organ]))


         ## Validation
         total_steps=len(val_dataloader)
         for i, data in enumerate(val_dataloader):
             if i>50:
                 break
             model.eval()
             images = data['image'].double()
             masks = data['mask'].type(torch.LongTensor)
             masks = masks.unsqueeze(1)
             images = images.to("cuda:1")
             masks = masks.to("cuda:1")

             outputs = model(images, data['task'])
             outputs = outputs.to("cuda:1")

             loss = criterion(outputs.double(), masks[:,0,:,:])
             print(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )
             logging.info('current task: ' + sample['task'][0])
             logging.info(f"Validation Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}", )

             dices = dice(outputs, masks, data['num_classes'][0], settings)
             background_dice = dices[1]
             organ_dice = dices[2]

             # Update data dic for relevant organ
             data_dic['Validation']['organ_dice']['cur'][sample['task'][0]].append(organ_dice)
             data_dic['Validation']['bg_dice']['cur'][sample['task'][0]].append(background_dice)
             data_dic['Validation']['CE']['cur'][sample['task'][0]].append(loss.item())

             # Update data dic [all organ]
             data_dic['Validation']['organ_dice']['cur']['all_organs'].append(organ_dice)
             data_dic['Validation']['bg_dice']['cur']['all_organs'].append(background_dice)
             data_dic['Validation']['CE']['cur']['all_organs'].append(loss.item())

         # Update data_dic['total_epochs']
         for l in loss_type:
            for organ in organs:
                data_dic['Validation'][l]['total_epochs'][organ].append(np.mean(data_dic['Training'][l]['cur'][organ]))



         print('End of epoch {} / {} \t Time Taken: {} min'.format(epoch, num_epochs,(time.time() - epoch_start_time) / 60))
         print('train loss: {} val_loss: {}'.format(np.mean(data_dic['Training']['CE']['cur']['all_organs']),
                np.mean(data_dic['Validation']['CE']['cur']['all_organs'])))
         print('train organ dice: {}  train background dice: {} val organ dice: {}  val background dice: {}'.format(
            np.mean(data_dic['Training']['organ_dice']['cur']['all_organs']),
            np.mean(data_dic['Training']['bg_dice']['cur']['all_organs']),
            np.mean(data_dic['Validation']['organ_dice']['cur']['all_organs']),
            np.mean(data_dic['Validation']['bg_dice']['cur']['all_organs'])))


    #plot_graph(num_epochs,settings,data_dic)


if __name__ == '__main__':
    json_path=r'G:\Deep learning\Datasets_organized\small_dataset\Experiments'
    start_exp_ind = 1
    num_exp = 8 # number of exp. to run

    for exp_ind in range(num_exp):
        exp_ind += start_exp_ind
        print('Start with experiment: {}'.format(exp_ind))
        with open(os.path.join(json_path,'exp_{}/exp_{}.json'.format(exp_ind, exp_ind))) as json_file:
            setting_dict = json.load(json_file)
        train(setting_dict)
