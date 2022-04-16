import pydicom
import numpy as np



def dicom_to_tensor(img_path, transforms, out_channels):
    orig_image = pydicom.read_file(img_path).pixel_array
    if out_channels > 1:
        img = np.dstack([orig_image for i in range(self.classifier.dataset.out_channels)])
        img = transforms(image=img)['image']
        img = torch.from_numpy(img)
        img = torch.moveaxis(img, -1, 0)
    else:
        img = transforms(image=orig_image)['image']
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    return img.float()

def dicom_to_hu(dicom_data):
    '''
    Converts DICOM absolute values to Hounsfield Units (CT Only)
    '''
    return (dicom_data.pixel_array*dicom_data.RescaleSlope+dicom_data.RescaleIntercept)

def wl_arr(arr, WW, WL):
    '''
    Returns a numpy array windowed and leveled to certain window and level values.
    '''
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(arr.copy(), lower, upper)
    return X

def dicom_image_processor(file_path, out_channels=None, WW=None, WL=None):
    '''
    Processes DICOM images. Uses a DICOM file and returns tuple of numpy array, image minimum value and image maximum value
    '''
    data = pydicom.read_file(file_path)
    modality = data.Modality
    in_channels = data.SamplesPerPixel
    if out_channels == None:
        out_channels = in_channels
    if modality == 'CT':
        arr = dicom_to_hu(data)
        if out_channels == 1:
            if all(i != None for i in [WW, WL]):
                img = wl_arr(arr, WW, WL)
            else:
                img = arr
        elif out_channels == 3:
            if all(i != None for i in [WW, WL]):
                channels = [wl_arr(arr, WW=WW[c], WL=WL[c]) for c in range(out_channels)]
                img = np.dstack(channels)
            else:
                channels = [arr for c in range(out_channels)]
                img = np.dstack(channels)
    else:
        arr = data.pixel_array
        if out_channels == 3:
                channels = [arr for c in range(out_channels)]
                img = np.dstack(channels)
                # print (img.shape)
                # img = np.moveaxis(img, 2, 0)
                # print (img.shape)
        else:
            img = arr
    img = img.astype(np.float)
    min = np.min(img)
    max = np.max(img)
    return img, min, max
