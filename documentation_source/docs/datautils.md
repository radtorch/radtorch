# Data Module <small> radtorch.datautils </small>


## list_of_files


    datautils.list_of_files(root)


!!! quote ""

    Create a list of file paths from a root folder and its sub directories.

      **Arguments**

      - root: _(str)_ path of target folder.

      **Output**

      - list of file paths.


      **Example**

        root_path = 'root/'
        list_of_files(root_path)

      <!-- **** -->

        ['root/folder1/0000.dcm', 'root/folder1/0001.dcm', 'root/folder2/0000.dcm', ...]


## path_to_class

    datautils.path_to_class(filepath)

!!! quote ""

    Creates a class name from the immediate parent folder of a target file.

      **Arguments**

      - filepath: _(str)_ path to target file.

      **Output**

      - _(str)_ folder name / class name.


      **Example**

        file_path = 'root/folder1/folder2/0000.dcm'
        path_to_class(file_path)

      <!-- **** -->

        'folder2'    



## root_to_class

    datautils.root_to_class(root)

!!! quote ""

    Creates list of classes and dictionary of classes and idx in a given data root. All first level subfolders within the root are converted into classes and given class id.


      **Arguments**

      - root: _(str)_ path of target root.

      **Output**

      - _(tuple)_ of
        - classes: _(list)_ of generated classes,
        - class_to_idx: _(dictionary)_ of classes and class id numbers


      **Example**

      This example assumes that root folder contains 3 folders (folder1, folder2 and folder3) each contains images of 1 class.

        root_folder = 'root/'
        root_to_class(root_folder)

      <!-- **** -->

        ['folder1', 'folder2', 'folder3'], {'folder1':0, 'folder2':1, 'folder3':2}


## class_to_idx
    datautils.class_to_idx(classes)

!!! quote ""
      Creates a dictionary of classes to classes idx from provided list of classes

      **Arguments**

      - classes: _(list)_ list of classes

      **Output**

      - Output: _(dictionary)_ dictionary of classes to class idx


      **Example**
      
        class_list = ['class1','class4', 'class2', 'class3']
        class_to_idx(class_list)

      <!-- **** -->

        {'class1':0, 'class2':1, 'class3':2, 'class4':3}

## dataset_from_folder

    datautils.dataset_from_folder(data_directory, is_dicom=True, mode='RAW',
                        wl=None, trans=Compose(ToTensor()))

!!! quote ""    
    Creates a dataset from a root directory using subdirectories as classes/labels.

    **Parameters**

    - data_director: _(str)_ target data root directory.

    - is_dicom: _(boolean)_ True for DICOM images, False for regular images.(default=True)

    - mode: _(str)_ output mode for DICOM images only. options: RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].

    - wl: _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN. In the form of : [[Level,Width], [Level,Width],…]. Only 3 combinations are allowed for MWIN (for now). (default=None)

    - trans: _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)


    **Methods**

    - **class_to_idx**

          Returns dictionary of dataset classes and corresponding class id.

    - **classes**

          Returns list of dataset classes

    - **info**

          Returns detailed information of the dataset.





## dataset_from_table

    datautils.dataset_from_table(data_directory, is_csv=True, is_dicom=True,
                                input_source=None, img_path_column='IMAGE_PATH',
                                img_label_column='IMAGE_LABEL', mode='RAW', wl=None,
                                trans=Compose(ToTensor()))          

!!! quote ""

    Creates a dataset using labels and filepaths from a table which can be either a excel sheet or pandas dataframe.


    **Parameters**

    - data_directory: _(str)_ target data directory.

    - is_csv: _(boolean)_ True for csv, False for pandas dataframe. (default=True)

    - is_dicom: _(boolean)_ True for DICOM images, False for regular images.(default=True)

    - input_source: _(str or pandas dataframe object)_ source for labelling data. This is path to csv file or name of pandas dataframe if pandas to be used.

    - img_path_column: _(str)_  name of the image path column in data input. (default = "IMAGE_PATH")

    - img_label_column: _(str)_  name of label column in the data input (default = "IMAGE_LABEL")

    - mode: _(str)_  output mode for DICOM images only. options: RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].

    - wl: _(list)_  list of lists of combinations of window level and widths to be used with WIN and MWIN.In the form of : [[Level,Width], [Level,Width],…]. Only 3 combinations are allowed for MWIN (for now).  (default=None)

    - transforms: _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)


    **Methods**

    - **class_to_idx**

          Returns dictionary of dataset classes and corresponding class id.

    - **classes**

          Returns list of dataset classes

    - **info**

          Returns detailed information of the dataset.    
