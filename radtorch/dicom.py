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
    '''
    .. include:: ./documentation/docs/dicomutils.md##window_dicom
    '''

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
    .. include:: ./documentation/docs/dicomutils.md##dicom_to_narray
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
    .. include:: ./documentation/docs/dicomutils.md##dicom_to_pil
    """

    ds = pydicom.read_file(filepath)
    pixels = ds.pixel_array
    pil_image = Image.fromarray(np.rollaxis(pixels, 0,1))
    return pil_image