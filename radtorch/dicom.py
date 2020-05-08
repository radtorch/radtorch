# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

from radtorch.settings import *


def window_dicom(filepath, level, width):
    
    """
    Description
    -----------
    Converts a DICOM image to HU and applies desired window/level

    Paramaters
    -----------
    filepath (string, required): path to DICOM image.
    level (integer, required): value of Level.
    width (integer, required): value of Window.

    Returns
    -----------
    numpy array of windowed DICOM image.
    """ 
    
    ds = pydicom.read_file(filepath)
    pixels = ds.pixel_array
    if ds.Modality == 'CT':
        img_hu = pixels*ds.RescaleSlope + ds.RescaleIntercept
        lower = level - (width / 2)
        upper = level + (width / 2)
        img_hu[img_hu<=lower] = lower
        img_hu[img_hu>=upper] = upper
        return img_hu
    else:
        return (pixels)

def dicom_to_narray(filepath, mode='RAW', wl=None):
    
    """
    Description
    -----------
    Converts a DICOM image to numpy array.

    Paramaters
    -----------
    filepath (string, required): path to DICOM image.
    mode (string, required): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level.
    wl (tuple or list of tuples, required): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)].

    Returns
    -----------
    numpy array of windowed DICOM image.
    """     
    
    if mode == 'RAW':
        ds = pydicom.read_file(filepath)
        img = ds.pixel_array
        return img
    elif mode == 'HU':
        ds = pydicom.read_file(filepath)
        pixels = ds.pixel_array
        if ds.Modality == 'CT':
            hu_img = pixels*ds.RescaleSlope + ds.RescaleIntercept
        else:
            hu_img = pixels
        return hu_img
    elif mode == 'WIN':
        if wl==None:
            print ('Error! argument "wl" cannot be empty when "WIN" mode is selected')
        elif len(wl) != 1:
            print ('Error! argument "wl" can only accept 1 combination of W and L when "WIN" mode is selected')
        else:
            win_img = window_dicom(filepath, wl[0][0], wl[0][1])
            return win_img
    elif mode == 'MWIN':
        if wl==None:
            print ('Error! argument "wl" cannot be empty when "MWIN" mode is selected')
        elif len(wl)!=3:
            print ('Error! argument "wl" must contain 3 combinations of W and L when "MWIN" mode is selected')
        else:
            img0 = window_dicom(filepath, wl[0][0], wl[0][1]).astype('int16')
            img1 = window_dicom(filepath, wl[1][0], wl[1][1]).astype('int16')
            img2 = window_dicom(filepath, wl[2][0], wl[2][1]).astype('int16')
            mwin_img = np.stack((img0,img1,img2), axis=-1)
            return mwin_img

def dicom_to_pil(filepath):

    """
    Description
    -----------
    Converts a DICOM image to PIL image object

    Paramaters
    -----------
    filepath (string, required): path to DICOM image.

    Returns
    -----------
    PIL image object
    """ 

    ds = pydicom.read_file(filepath)
    pixels = ds.pixel_array
    pil_image = Image.fromarray(np.rollaxis(pixels, 0,1))
    return pil_image

# 3D-Slicer Functions
def find_coordinates(vol="inputvol",roi="croproi"):
    roiNode = getNode(roi)
    volNode = getNode(vol)
    bounds = [0,0,0,0,0,0]
    mat = vtk.vtkMatrix4x4()
    volNode.GetRASToIJKMatrix(mat)
    roiNode.GetRASBounds(bounds)
    p1 = [bounds[1], bounds[3], bounds[5], 1]
    p2 = [bounds[0], bounds[2], bounds[4], 1]
    i1 = mat.MultiplyFloatPoint(p1)
    i2 = mat.MultiplyFloatPoint(p2)
    coordinates = [i1[0], i2[0], i1[1], i2[1], i1[2], i2[2]]
    coordinates = [int(x) for x in coordinates]
    print (coordinates)

# Split Multiphasic study
def split_multiphasic_scan(input_dir, output_dir, modality=''):
    if modality=='MRI':
        files = []
        position = []
        instance = []
        print ('Creating file list.')
        for r, d, f in os.walk(input_dir):
            for i in f:
                i_path = os.path.join(r,i)
                files.append(i)
                ds = pydicom.read_file(i_path, force=True)
                position.append(ds.ImagePositionPatient[2])
                instance.append(ds.InstanceNumber)
        df = pd.DataFrame({'ImageId':files, 'ImagePositionPatient':position, 'InstanceNumber':instance})
        number_phases = len(files) / len(df['ImagePositionPatient'].unique())
        print ('Calculating number of phases.')
        if number_phases.is_integer():
            number_phases = int(number_phases)
        else:
            print ('Error. Number of images can not be split across the calculated number of phases')
            return
        phases = []
        for i in range(0,number_phases):
            phases.append('phase_'+str(i))
        phases_list = phases * len(df['ImagePositionPatient'].unique())
        df = df.sort_values(by=['ImagePositionPatient', 'InstanceNumber']);
        df['phase'] = phases_list
        print ('Splitting into',number_phases,'phases.')
        for i, r in tqdm(df.iterrows(), total=len(df)):
            img_path = input_dir+r['ImageId']
            ds = pydicom.read_file(img_path)
            ds.SeriesDescription = ds.SeriesDescription+'_'+r['phase']
            ds.SeriesInstanceUID = ds.SeriesInstanceUID+r['phase'][-1]
            ds.SeriesNumber =  str(ds.SeriesNumber)+r['phase'][-1]
            ds.save_as(output_dir+r['ImageId'])
        print ('Splitting completed successfully')

    elif modality=='CT':
        files = []
        slice_location = []
        aq_time = []
        print ('Creating file list.')
        for r,ds, fs in os.walk(input_dir):
            for f in fs:
                img_path = input_dir+f
                dicom_data = pydicom.read_file(img_path)
                files.append(f)
                slice_location.append(dicom_data.SliceLocation)
                aq_time.append(dicom_data.AcquisitionTime)
        dataframe = pd.DataFrame({'img_id':files, 'slice_location':slice_location, 'aq_time':aq_time})
        dataframe.sort_values(by=['aq_time', 'slice_location'])
        print ('Calculating number of phases.')
        unique_aq_time = dataframe.aq_time.unique().tolist()
        print ('Splitting into',len(unique_aq_time),'phases.')
        for i, r in tqdm(dataframe.iterrows(), total=len(dataframe)):
            img_path = input_dir+r['img_id']
            ds = pydicom.read_file(img_path)
            ds.SeriesDescription = ds.SeriesDescription+'_'+ str(unique_aq_time.index(r['aq_time']))
            ds.SeriesInstanceUID = ds.SeriesInstanceUID+ str(unique_aq_time.index(r['aq_time']))
            ds.SeriesNumber =  str(ds.SeriesNumber)+str(unique_aq_time.index(r['aq_time']))
            ds.save_as(output_dir+r['img_id'])
        print ('Splitting completed successfully')
