import os, pydicom, torch, torchvision, glob, scipy.ndimage
import numpy as np
import pandas as pd

import SimpleITK as sitk
from PIL import Image
from pathlib import Path


from .general import *
from .const import *
from .dicom import *



def find_classes(directory):
    '''
    Finds classes from folder names in a parent directory
    '''
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def check_zero_image(table, path_col='path'):
    zero_img = []
    for i, r in table.iterrows():
        if np.max(pydicom.read_file(r[path_col]).pixel_array) == 0:
            zero_img.append(True)
        else:
            zero_img.append(False)
    table['zero_img'] = zero_img
    return table[table['zero_img']==False]


def dicom_images_to_table(folder, extension='dcm', path_col='path', label_col='label'):
    '''
    Creates a table of image path and corresponding labels
    '''
    df = pd.DataFrame()
    list_path = []
    list_label =[]
    for file in glob.glob(folder + "**/*."+extension, recursive=True):
        list_path.append(file)
        list_label.append(Path(os.path.join(folder, file)).parent.name)
    df[path_col] = list_path
    df[label_col] = list_label
    return df


def dicom_volume_to_table(folder, extension='dcm', path_col='path', label_col='label', study_col='study_id', use_vol_file=False):
    '''
    input parent folder > output class list, class_to_idx dictionary and series table
    '''
    classes, class_to_idx = find_classes(folder)
    table = pd.DataFrame(columns=[study_col, path_col, "num_images", label_col])
    if use_vol_file:
        table = pd.DataFrame(columns=[study_col, path_col, "num_images", "H", "W", label_col])
    else:
        table = pd.DataFrame(columns=[study_col, path_col, "num_images", label_col])

    for c in classes:
        study_dir = os.path.join(folder, c)
        study_idx = [x for x in os.walk(os.path.join(folder, c))][0][1]
        study_paths = [x[0] for x in os.walk(os.path.join(folder, c))][1:]
        if use_vol_file:
            # table = pd.DataFrame(columns=[study_col, path_col, "num_images", "H", "W", label_col])
            for i in range(len(study_idx)):
                vol_path = path_fix(study_paths[i])+[y for y in [x[2] for x in os.walk(study_paths[i])][0] if y.endswith('.pt')][0]
                vol = torch.load(vol_path)
                # table.loc[len(table.index)] = [study_idx[i],vol_path,vol.shape[1],c]
                table.loc[len(table.index)] = [study_idx[i],vol_path,vol.shape[1],vol.shape[2],vol.shape[3],c]

        else:
            for i in range(len(study_idx)):
                table.loc[len(table.index)] = [study_idx[i],study_paths[i],len([file for file in glob.glob(study_paths[i] + "/" + "**/*." + extension, recursive=True)]),c,]
    return classes, class_to_idx, table


def directory_to_tensor(directory, extension='dcm', transforms=None, out_channels=1, WW=None, WL=None):
    '''
    input folder > output 4D tensor (channels, depth, H, W)
    '''
    directory=path_fix(directory)
    file_list = [directory+i for i in os.listdir(directory) if i.endswith(extension)]
    assert len(file_list) >1, 'Error: Not more than one image was found in directory: '+directory
    file_list = sorted ([i for i in file_list],key=lambda x: (pydicom.read_file(x)).SliceLocation, reverse=True)
    dcm_list = [pydicom.read_file(i) for i in file_list]
    volume = np.stack([image_to_tensor(p, out_channels, transforms, WW, WL).numpy() for p in file_list])
    volume = torch.from_numpy(volume).moveaxis(0,1)
    original_spacing = [float(dcm_list[0].PixelSpacing[0]), float(dcm_list[0].PixelSpacing[1]), dcm_list[0].SliceLocation - dcm_list[1].SliceLocation]
    min = torch.min(volume)
    max = torch.max(volume)
    return volume, min, max, original_spacing


def resample_dicom_volume(volume, original_spacing, resample_spacing=[-1, -1, -1], resample_slices=None):
    '''
    input numpy array > ouput numpy array

    Code modified from:
    https://github.com/rachellea/ct-volume-preprocessing/blob/master/preprocess_volumes.py

    Resamples a 4d numpy array/tensor of dicom data into a new 4d tensor ready for Conv3d.
    The new array can either have a custom spacing or custom number of images.
    If custom number of images is desired, the output spacing will be automatically determined.
    '''
    orig_shape = volume.shape
    if torch.is_tensor(volume):
        volume = volume.moveaxis(0,1)
        volume = volume.numpy()

    for index, value in enumerate(resample_spacing):
        if value == -1:
            resample_spacing[index] = original_spacing[index]

    if resample_slices:
        resample_spacing[2] = volume.shape[0] * original_spacing[2] / resample_slices

    ctvol_itk = sitk.GetImageFromArray(volume)
    ctvol_itk.SetSpacing(original_spacing)
    original_size = ctvol_itk.GetSize()
    out_shape = [int(np.round(original_size[0] * (original_spacing[0] / resample_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / resample_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / resample_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(resample_spacing)
    resample.SetSize(out_shape)
    resample.SetOutputDirection(ctvol_itk.GetDirection())
    resample.SetOutputOrigin(ctvol_itk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resampled_volume = resample.Execute(ctvol_itk)
    resampled_volume = sitk.GetArrayFromImage(resampled_volume)
    resampled_volume = torch.from_numpy(resampled_volume).moveaxis(0,1)
    return resampled_volume


def save_checkpoint(classifier, epochs=None, current_epoch=None, output_file=None):
    if classifier.type == 'torch':
        checkpoint = {'timestamp': current_time(),
                      'type':classifier.type,
                      'classifier':classifier,
                      'epochs':epochs,
                      'current_epoch':current_epoch,
                      'optimizer_state_dict' : classifier.optimizer.state_dict(),
                      'train_losses': classifier.train_losses,
                      'valid_losses': classifier.valid_losses,
                      'valid_loss_min': classifier.valid_loss_min,}
        if output_file == None:
            output_file = classifier.name+'epoch_'+str(current_epoch)+'.checkpoint'
    elif classifier.type == 'sklearn':
        checkpoint = {'timestamp': current_time(),
                      'type':classifier.type,
                      'classifier':classifier,
                      'model':classifier.best_model}
        if output_file == None:
            output_file = classifier.name+'.checkpoint'
    torch.save(checkpoint, output_file)


def load_checkpoint(classifier, checkpoint_path):
    classifier.checkpoint = torch.load(checkpoint_path)
    checkpoint_classifier = classifier.checkpoint['classifier']
    classifier.__dict__.update(checkpoint_classifier.__dict__)
    classifier.current_epoch = classifier.checkpoint['current_epoch']


def image_to_tensor(path,  out_channels=1, transforms=None, WW=None, WL=None): #OK

    img_type = os.path.splitext(path)[1]
    if img_type in dicom_extensions:
            img, min, max = dicom_image_processor(path, out_channels, WW, WL)
    else:
        img = Image.open(path).convert("RGB")
        img = np.asarray(img)

    if transforms:
        img = transforms(image=img)['image']
        img = torch.from_numpy(img)
        if out_channels > 1:
            img = torch.moveaxis(img, -1, 0) # Need to check the effect of that on non-DICOM images.
        else:
            img = img.unsqueeze(0)
    else:
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = transforms(img)
    # img = img.unsqueeze(0)
    return img.float()
