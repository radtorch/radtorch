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



def getDuplicatesWithCount(listOfElems):
    """
    .. image:: pass.jpg
    """

    dictOfElems = dict()
    for elem in listOfElems:
        if elem in dictOfElems:
            dictOfElems[elem] += 1
        else:
            dictOfElems[elem] = 1
    dictOfElems = { key:value for key, value in dictOfElems.items() if value > 1}
    return dictOfElems



def export(item, path):
    outfile = open(path,'wb')
    pickle.dump(item,outfile)
    outfile.close()


def log(msg):
    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    message='['+timestamp+']: '+msg
    print (message)
    file_operation=open(logfile, 'a')
    file_operation.write('\n')
    file_operation.write(message)
    file_operation.close()


def showlog():
    f = open(logfile, 'r')
    file_contents = f.read()
    print (file_contents)
    f.close()

def clearlog():
    open(logfile, 'w').close()


# def supported():
#     print ('Supported Network Architectures:')
#     for i in supported_models:
#         print (i)
#     print('')
#     print ('Supported Image Classification Loss Functions:')
#     for i in supported_image_classification_losses:
#         print (i)
#     print('')
#     print ('Supported Optimizers:')
#     for i in supported_optimizer:
#         print (i)
#     print ('')
#     print ('Supported non DICOM image file types:')
#     for i in IMG_EXTENSIONS:
#         print (i)
