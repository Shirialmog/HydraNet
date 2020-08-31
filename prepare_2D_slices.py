import os
from os import path
import numpy as np
import nibabel as nb
import csv
import matplotlib.pyplot as plt
import torch.utils.data as utils
from scipy import ndimage
import math
import imageio
import SimpleITK as sitk

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return ct_scan


def re_sample(slice, end_shape, order=3):
    zoom_factor = [n / float(o) for n, o in zip(end_shape, slice.shape)]
    if not np.all(zoom_factor == (1, 1)):
        data = ndimage.zoom(slice, zoom=zoom_factor, order=order, mode='constant')
    else:
        data = slice
    return data

def pre_process(slice,bottom_thresh,top_thresh): #receives a 2D image and intensity thresholds and performs windowing
    new_image = np.copy(slice)
    new_image[slice <= bottom_thresh] = bottom_thresh
    new_image[slice >= top_thresh] = top_thresh
    return(new_image)

def get_truncate_index(scan,num_slices,percent): #function that takes only some slices on z axis
    top_index=num_slices-1
    bottom_index=0

    for i in range (num_slices):
        slice = scan[:,:,i]
        result = sum(sum(slice))
        if result != 0:
            bottom_index = i
            break
    for i in range (num_slices-1,-1,-1):
        slice = scan[:, :, i]
        result = sum(sum(slice))
        if result != 0:
            top_index = i
            break

    num_good_slices = top_index - bottom_index + 1
    on_top = min(num_slices - top_index - 1, math.ceil(percent*num_good_slices))
    on_bottom = min(bottom_index, math.ceil(percent*num_good_slices))
    num_slices_each_side = min(on_top, on_bottom)

    final_top = top_index + num_slices_each_side
    final_bottom = bottom_index - num_slices_each_side

    return final_bottom,final_top

def make_binary(label):
    label[label!=0] = 1
    return None

def main(path, task_name,end_shape,truncate=False, binary=False):
    if not os.path.exists(save_path+'/'+task_name):
        os.mkdir(save_path+'/'+task_name, 777)

    #create csv for metadata
    meta_data = open(save_path+'/'+task_name + '/' + task_name+ '_metadata.csv' , mode='w')
    wr = csv.writer(meta_data, lineterminator='\n')

    for set in ['Training','Validation','Test']:
        files=os.listdir(path + '/' + set)
        print (files)
        new_path = save_path + '/' + task_name + '/' + set
        label_path = save_path + '/' + task_name  +'/' + set + '_Labels'
        os.mkdir(new_path, 777)
        os.mkdir(label_path, 777)

        for ind,file in enumerate(files,0):

            if task_name=="BRATS":
                # In each folder there are 5 scans: T1,T1ce,T2,FLAIR and label
                # create 2.5D slices that are made of T1,T1ce,T2
                phases=os.listdir(path+'/'+set+'/'+file)
                for phase in phases: #runs on all the different scans for each patient
                    if 't1.nii' in str(phase):
                        t1_scan=nb.load(path+'/'+set+'/'+file+'/'+phase)
                        t1_scan=t1_scan.get_data()
                    if 't1ce' in str(phase):
                        t1ce_scan=nb.load(path+'/'+set+'/'+file+'/'+phase)
                        t1ce_scan = t1ce_scan.get_data()
                    if 't2' in str(phase):
                        t2_scan=nb.load(path+'/'+set+'/'+file+'/'+phase)
                        t2_scan = t2_scan.get_data()
                    if 'seg' in str(phase):
                        label=nb.load(path+'/'+set+'/'+file+'/'+phase)
                        label=label.get_data()
                        make_binary(label)

                num_slices = t1_scan.shape[2]
                if truncate==True:
                    bottom_index, top_index = get_truncate_index(label, num_slices, 0.2)
                    print (bottom_index,top_index)
                    t1_scan = t1_scan[:, :, bottom_index:top_index]
                    t1ce_scan = t1ce_scan[:, :, bottom_index:top_index]
                    t2_scan = t2_scan[:, :, bottom_index:top_index]
                    label = label[:, :, bottom_index:top_index]

                num_slices = t1_scan.shape[2]
                print(t1_scan.shape)
                print(t1ce_scan.shape)
                print (t2_scan.shape)
                output = np.empty((3,end_shape[0], end_shape[1]), dtype=float, order='C')
                for i in range(num_slices-1):
                    # adding relevant data to csv:
                    # scan, number of slice, set(training/val/test), slice path, label path
                    wr.writerow([file, str(i), set, new_path + '/slice' + str(i), label_path + '/slice' + str(i)])
                    output_new=output
                    # create three slices from data and re samples them to wanted size, stack the three slices to form 2.5D slices
                    output_new[1,:, :] = np.rot90(re_sample(t1ce_scan[:, :, i], end_shape))  # middle slice
                    output_new[0,:, :] = np.rot90(re_sample(t1_scan[:, :, i - 1], end_shape) ) # bottom slice
                    output_new[2,:, :] = np.rot90(re_sample(t2_scan[:, :, i + 1], end_shape))  # top slice
                    label_new= np.rot90(re_sample(label[:, :, i], end_shape, order=1))

                    np.save(new_path + '/' + str(ind) + '_slice_' + str(i), output_new)
                    np.save(label_path + '/' + str(ind) + '_slice_' + str(i), label_new)


            else: ##not BRATS
                img = nb.load(path + '/' + set+'/'+file)
                label = nb.load(path + '/Labels' + '/' + file)

                data = img.get_data()
                header=img.header
                originalSpacing=header['pixdim'][1]
                spacingFactor=end_shape[0]/data.shape[0]
                newSpacing=originalSpacing*spacingFactor
                label = label.get_data()

                num_slices = data.shape[2]

                if task_name=='Prostate':
                    data=data[:,:,:,0]


                if truncate==True:
                    bottom_index,top_index = get_truncate_index(label,num_slices,0.2)
                    data = data[:, :, bottom_index:top_index]
                    label = label[:, :, bottom_index:top_index]

                if binary==True:
                    make_binary(label)

                print (data.shape)
                num_slices = data.shape[2]
                data = np.dstack((data[:, :, 0], data, data[:, :, num_slices - 1])) #padding the slices
                label = np.dstack((label[:, :, 0], label, label[:, :, num_slices - 1])) #padding the slices
                output = np.empty((3,end_shape[0],end_shape[1]), dtype=float, order='C')

                # create a stack of our "2.5D slices", each containing 3 slices

                for i in range(1, num_slices+1):
                    # adding relevant data to csv:
                    # scan, number of slice, set(training/val/test), spacing, slice path, label path
                    wr.writerow([file, str(i), set, str(newSpacing), new_path + '/slice' + str(i), label_path + '/slice' + str(i)])
                    output_new=output
                    output_new[1, :, :] = re_sample(data[:, :, i], end_shape)  # middle slice
                    output_new[0, :, :] = re_sample(data[:, :, i - 1], end_shape) # bottom slice
                    output_new[2, :, :] = re_sample(data[:, :, i + 1], end_shape)  # top slice
                    label_new = re_sample(label[:, :, i], end_shape, order=1)
                    np.save(new_path + '/' + str(ind) + '_slice_' + str(i), output_new)
                    np.save(label_path + '/' + str(ind) + '_slice_' + str(i), label_new)

    meta_data.close()
    return None
############################################

if __name__ == '__main__':
    path = r'G:\Deep learning\Datasets_organized\Uterus'  # change to relevant source path
    task_name = 'Uterus'
    save_path = r'G:\Deep learning\Datasets_organized\Prepared_Data'  # change to where you want to save data
    end_shape = (384, 384)  # wanted slice shape after resampling
    main(path,task_name,end_shape,truncate=True,binary=True)