import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import json
import logging
import sys
import torch.nn as nn
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from SegmentationModule.SegmentationSettings import SegSettings
import Models as models
import copy
from HydraTrain import pre_processing,clip,clip_n_normalize,zscore_normalize,create_augmentations


matplotlib.use('TkAgg')
cudnn.benchmark = True

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
        self.mask_labels_numeric = mask_labels_numeric
        self.mask_class_weights_dict = mask_class_weights_dict
        self.is_metric = is_metric
        self.eps = 1e-10
        self.tot_weight = torch.sum(torch.Tensor(list(mask_class_weights_dict.values()))).item()

    def forward(self, pred, target,task):
        if self.is_metric:
            if self.classes >1:
                pred = torch.argmax(pred, dim=1)
                pred = torch.eye(self.classes)[pred]
                pred = pred.transpose(1, 3).to(device)

                if task[0]==('lits' or 'prostate'):
                    pred_p = post_process(pred)
                    pred=torch.Tensor(pred_p).to(device)
                    pred=pred.unsqueeze(0)
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
        confusion_vector = pred / target.float()
        TP = torch.sum(confusion_vector[0, 1, :, :] == 1).item()
        FP = torch.sum(confusion_vector[0, 1, :, :] == float('inf')).item()
        TN = torch.sum(torch.isnan(confusion_vector[0, 1, :, :])).item()
        FN = torch.sum(confusion_vector[0, 1, :, :] == 0).item()
        eps=0.000001
        sensitivity=TP/(TP+FN+eps)
        specificity=TN/(TN+FP+eps)
        batch_intersection = torch.sum(pred * target.float(), dim=tuple(list(range(2, self.dimension + 2))))
        batch_union = torch.sum(pred, dim=tuple(list(range(2, self.dimension + 2)))) + torch.sum(target.float(),dim=tuple(list(range(2,self.dimension + 2))))
        tumour_dice=None
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
                return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),sensitivity,specificity,tumour_dice.mean().item()],pred

        if self.is_metric:
            return [mean_dice_val.mean().item(), background_dice.mean().item(), organ_dice.mean().item(),sensitivity,specificity],pred
        else:
            return -mean_dice_val

def save_samp_test(image,mask,task,prediction,iter,loss):
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[1, :, :], cmap="gray")
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title('Mask(GT)')

    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = np.squeeze(prediction)
    plt.subplot(1, 3, 3)
    plt.imshow(np.fliplr(np.rot90(prediction,3)), cmap="gray")
    plt.title('Prediction')
    plt.suptitle('Task: ' + task + ' Epoch: '+  ' Iteration: ' + str(iter) + ' Loss: '+ str(loss))
    plt.tight_layout()
    fig.savefig(os.path.join(settings.test_dir,task,'test_it_{}.{}'.format(iter, 'png')))
    plt.close('all')


def dice(pred, target, num_classes,settings,task):
    if num_classes==2:
        mask_labels={'background': 0,  'organ': 1}
        loss_weights= {'background': 1, 'organ': 1}
    elif num_classes==3: ## pancreas only
        mask_labels = {'background': 0, 'organ': 1, 'tumour':2} ##check if this is true
        loss_weights = {'background': 1, 'organ': 10,'tumour':20}
    dice_measurement = DiceLoss(classes=num_classes,
                               dimension=settings.dimension,
                               mask_labels_numeric=mask_labels,
                               mask_class_weights_dict=loss_weights,
                               is_metric=True)
    [*dices] = dice_measurement(pred, target,task)
    return dices

def post_process(seg):
    seg_morph =seg[:,1,:,:].clone().detach().cpu()
    organ = seg_morph>0
    kernel_morph = np.ones((1, 15, 15))
    organ=binary_erosion(organ, structure=kernel_morph, iterations=1)
    organ = binary_dilation(organ, structure=kernel_morph, iterations=1)
    organ = binary_fill_holes(organ)

    bg=np.ones(organ.shape)
    bg=bg-organ
    bg=torch.Tensor(bg)
    organ=torch.Tensor(organ)
    final=torch.cat((bg,organ),0)
    return final

def test(settings, exp_ind,model,device):
    model.to(device)
    model = model.double()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,10]).double().to(device))
    print ('model loaded')

    # Initialize 'data_dic', nested dictionary, will contain all losses and dice for all organs
    organs = ['all_organs', 'spleen', 'prostate', 'lits', 'brain', 'pancreas', 'hepatic_vessel', 'left_atrial']
    count_type = ['cur', 'final']
    loss_type = ['CE', 'bg_dice', 'organ_dice','mean_dice','sensitivity','specificity']

    partition_by_organ = dict(zip(organs, [list() for i in organs]))  # first dic - organs
    partition_by_count = dict(zip(count_type, [copy.deepcopy(partition_by_organ) for i in count_type]))
    data_dic = dict(zip(loss_type, [copy.deepcopy(partition_by_count) for i in loss_type]))

    ## load data
    test_dataset_list = []
    for organ in organs[1:]:
        organ_test_dataset = Seg_Dataset(organ, settings.definition_file_dir + '/' + organ + '/Test',
                                          settings.definition_file_dir + '/' + organ + '/Test_Labels', 2, settings)
        test_dataset_list.append(organ_test_dataset)

    test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print ('Data ready')

    total_steps=len(test_dataloader)
    print('Testing... ')
    for i, data in enumerate(test_dataloader):
        torch.no_grad()

        images = data['image'].double()
        masks = data['mask'].type(torch.LongTensor)
        masks = masks.unsqueeze(1)
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images, data['task'])
        outputs = outputs.to(device)
        loss = criterion(outputs.double(), masks[:,0,:,:])

        one_hot = torch.DoubleTensor(masks.size(0), data['num_classes'][0], masks.size(2), masks.size(3)).zero_()
        one_hot = one_hot.to(device)
        masks_dice = one_hot.scatter_(1, masks.data, 1)

        dices,pred = dice(outputs, masks_dice, data['num_classes'][0], settings,data['task'])
        mean_dice = dices[0]
        background_dice = dices[1]
        organ_dice = dices[2]
        sensitivity=dices[3]
        specificity=dices[4]
        if len(dices) == 6:
            tumour_dice = dices[5]

        if i % 20 == 0:
            save_samp_test(data['image'][0], data['mask'][0], data['task'][0], pred, i,
                                 organ_dice)
        ##update dict
        data_dic['CE']['cur']['all_organs'].append(loss.item())
        data_dic['organ_dice']['cur']['all_organs'].append(organ_dice)
        data_dic['bg_dice']['cur']['all_organs'].append(background_dice)
        data_dic['sensitivity']['cur']['all_organs'].append(sensitivity)
        data_dic['specificity']['cur']['all_organs'].append(specificity)

        data_dic['organ_dice']['cur'][data['task'][0]].append(organ_dice)
        data_dic['bg_dice']['cur'][data['task'][0]].append(background_dice)
        data_dic['CE']['cur'][data['task'][0]].append(loss.item())
        data_dic['sensitivity']['cur'][data['task'][0]].append(sensitivity)
        data_dic['specificity']['cur'][data['task'][0]].append(specificity)


        print('Iteration: {}/{} Test loss: {} '.format(i, total_steps,np.mean(data_dic['CE']['cur']['all_organs'])))
        print('Test organ dice: {}  Test background dice: {}'.format(
            np.mean(data_dic['organ_dice']['cur']['all_organs']), np.mean(data_dic['bg_dice']['cur']['all_organs'])))


    for l in loss_type:
        for organ in organs:
            data_dic[l]['final'][organ].append(np.mean(data_dic[l]['cur'][organ]))

    for l in loss_type:
        for organ in organs:
            print ('{} {}: '.format(organ, l), data_dic[l]['final'][organ] )

    ## save results as text file 
    with open(r'G:\Deep learning\Datasets_organized\small_dataset\Experiments\exp_{}\test_results_exp_{}_epoch_{}.txt'.format(exp_ind,exp_ind,epoch),'w') as f:
        a=''
        for l in loss_type:
            for organ in organs:
                a+='{} {}: '.format(organ, l)+ str(data_dic[l]['final'][organ])+'\n'
        f.writelines(a)


if __name__ == '__main__':
    exp_ind=2
    epoch=8
    device='cuda:1'
    #open current exp settings
    with open(r'G:\Deep learning\Datasets_organized\small_dataset\Experiments\exp_{}\exp_{}.json'.format(exp_ind,exp_ind)) as json_file:
        setting_dict = json.load(json_file)
        settings = SegSettings(setting_dict, write_logger=True)
    #create a model with the loaded settings
    model = models.Unet_2D(encoder_name=settings.encoder_name,
                           encoder_depth=settings.encoder_depth,
                           encoder_weights=settings.encoder_weights,
                           decoder_use_batchnorm=settings.decoder_use_batchnorm,
                           decoder_channels=settings.decoder_channels,
                           in_channels=settings.in_channels,
                           classes=settings.classes,
                           activation=settings.activation)
    # load the weights of the last epoch
    model.load_state_dict(torch.load('G:\Deep learning\Datasets_organized\small_dataset\Experiments\exp_{}\checkpoint\exp_{}_epoch_{}.pt'.format(exp_ind,exp_ind,epoch)))
    test_folder=r'G:\Deep learning\Datasets_organized\Prepared_Data'

    test(settings, exp_ind,model,device)
