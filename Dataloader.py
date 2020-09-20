from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib
from torchvision import transforms

class CoronaDatasetTrain(Dataset):
    """
    torch dataLoader for masks or images
    """

    def __init__(self, settings, images_list, transforms, binary_label=False):
        """
        Initialize this dataset class. for train mode
        """
        self.transform = transforms
        self.data_images_list = images_list
        self.settings = settings
        self.binary_label = binary_label

    def __len__(self):
        return len(self.data_images_list)

    def __getitem__(self, idx):
        image_path = self.data_images_list[idx]
        mask_path = image_path.replace('ct', 'seg').replace('data', 'labels')
        if 'rp' not in image_path:
            ind = image_path.split('\\')[-1].split('.')[1]
            ind = '0.{}'.format(ind)
            mask_path = mask_path.replace(ind, '0')
        image = np.load(image_path) # image is (384, 384, 3) and the relevant slice is the middle one
        case = image_path.split('\\')[-1]
        score = float(case.split('_')[-1].split('.')[0] + '.' + case.split('_')[-1].split('.')[1])

        if self.transform:
            image = self.clip_n_normalize(image)
            image = self.transform(image)
        else:
            tensor_transform = transforms.ToTensor()
            image = self.clip_n_normalize(image)
            image = tensor_transform(image)

        mask = np.load(mask_path).astype('uint8')
        if self.binary_label:
            if 'rp_im' in mask_path:
                mask[mask > 0] = 1
            else:
                mask[mask == 1] = 0
                mask[mask == 2] = 0
                mask[mask == 3] = 1

        if self.transform:
            mask = self.transform(mask)
        else:
            # transform = transforms.Compose([transforms.ToTensor()])
            mask = torch.tensor(mask)

        sample = {'image': image.float(),
                  'mask': mask,
                  'score': score}

        return sample

    def clip_n_normalize(self, data):
        if self.settings.clipping:
            # clip and normalize
            min_val = self.settings.min_clip_val
            max_val = self.settings.max_clip_val
            data = (data - min_val) / (max_val - min_val)
            data[data > 1] = 1
            data[data < 0] = 0
        else:
            # only normalize
            data = (data - np.min(data)) / np.ptp(data)
        return data
