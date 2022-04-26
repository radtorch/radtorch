import torch, uuid,  torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from copy import deepcopy
from torch.utils.data.dataset import Dataset
from mpl_toolkits.axes_grid1 import ImageGrid


from .utils import *


class ImageObject(): #OK
    """
    Creates a 3D tensor whose dimensions = [channels, width, height] from an image path.

    Args:

      path (str): Path to an image.
      out_channels (int, optional): Number of output channels. Only 1 and 3 channels supported.
      transforms (list, optional): Albumentations transformations. See [Image Augmentation](https://albumentations.ai/docs/getting_started/image_augmentation/).
      WW (int or list, optional): Window width for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
      WL (int or list, optional): Window level for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).

    Returns:

      tensor: 3D tensor whose dimensions = [channels, width, height]

    Examples:

        ```python
        >>> i = radtorch.data.ImageObject(path='data/PROTOTYPE/DIRECTORY/abdomen/abd_1/1-001.dcm')
        >>> i.shape

        torch.Size([1, 512, 512])
        ```

    """
    def __new__(cls, path,  out_channels=1, transforms=None, WW=None, WL=None, **kwargs):
        return image_to_tensor(path, out_channels, transforms, WW, WL)


class ImageDataset(Dataset): #OK

    """
    Creates pytorch dataset(s) and dataloader(s) objects from a parent folder. Use this class for image tasks that invovles handling each single image as a single instance of your dataset.

    Examples:

        ```python
        import radtorch
        import albumentations as A

        # Specify image transformations
        T = A.Compose([A.Resize(256,256)])

        # Create dataset object
        ds = radtorch.data.ImageDataset(
                                        folder='data/4CLASS/',
                                        split={'valid':0.2, 'test':0.2},
                                        out_channels=1,
                                        transforms={'train':T,'valid': T,'test': T},
                                         )

        ds.data_stat()
        ```
        <div style="text-align:center"><img src="../assets/5.png" style="width:50%; height:50%" /></div>

        ```python
        ds.table
        ```
        <div style="text-align:center"><img src="../assets/6.png" style="width:50%; height:50%" /></div>


    Parameters:

        folder (str): Parent folder containing images. `radtorch.ImageDataset` expects images to be arranged in the following structure:
            ```
            root/
                class_1/
                        image_1
                        image_2
                        ...
                class_2/
                        image_1
                        image_2
                        ...
            ```

        name (str, optional): Name to be give to the dataset. If none provided, the current date and time will be used to created a generic dataset name. (default=None)
        label_table (str|dataframe, optional): The table containing data labels for your images. Expected table should contain at least 2 columns: image path column and a label column. Table can be string path to CSV or a pandas dataframe object.(default=None)
        instance_id (bool, optional): True if the data provided in the image path column in label_table contains the image id not the absolute path for the image. (default= False)
        add_extension (bool, optional): If instance_id =True, use this to add extension to image path as needed. Extension must be provided without "." e.g. "dcm". (default=False)
        out_channels (int, optional): Number of output channels. (default=1)
        WW (int or list, optional): Window width for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
        WL (int or list, optional): Window level for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
        path_col (str, optional): Name of the column containing image path data in the label_table. (default='path')
        label_col (str, optional): Name of the column containing label data in the label_table. (default='label')
        extension (str, optional): Type/Extension of images. (default='dcm')
        transforms (dict, optional): Dictionary of Albumentations transformations in the form of {'train': .. , 'valid': .. , 'test': .. }. See https://albumentations.ai/docs/getting_started/image_augmentation/ . (default=None)
        random_state (int, optional): Random seed (default=100)
        sample (float, optional): Sample or percent of the overall data to be used. (default=1.0)
        split (dict): dictionary defining how data will be split for training/validation/testing. Follows the sturcture {'valid': float, 'test': float} or {'valid':'float'} in case no testing subset is needed. The percent of the training subset is infered automatically.
        ignore_zero_img (bool, optional): True to ignore images containig all zero pixels. (default=False)
        normalize (bool, optional): True to normalize image data between 0 and 1. (default=True)
        batch_size (int, optional): Dataloader batch size. (default = 16)
        shuffle (bool, optional): True to shuffle images during training. (default=True)
        weighted_sampler (bool, optional): True to use a weighted sampler for unbalanced datasets. See https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler. (default=False)
        num_workers (int, optional): Dataloader CPU workers. (default = 0)


    Attributes:
        classes (list): List of generated classes/labels.
        class_to_idx (dict): Dictionary of generated classes/labels and corresponding class/label id.
        idx_train (list): List of index values of images/instances used for training subset. These refer to index of `ImageDataset.table`.
        idx_valid (list):List of index values of images/instances used for validation subset. These refer to index of `ImageDataset.table`.
        idx_test (list): List of index values of images/instances used for testing subset. These refer to index of `ImageDataset.table`.
        table (pandas dataframe): Table of images , paths and their labels.
        table_train (pandas dataframe): Table of images used for training. Subset of `ImageDataset.table`.
        table_valid (pandas dataframe): Table of images used for validation. Subset of `ImageDataset.table`.
        table_test (pandas dataframe): Table of images used for testing. Subset of `ImageDataset.table`.
        tables (dict): Dictionary of all generated tables in the form of: {'train': table, 'valid':table, 'test':table}.
        dataset_train (pytorch dataset object): Training [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        dataset_valid (pytorch dataset object): Validation [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        dataset_test (pytorch dataset object): Testing [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        datasets (dict): Dictionary of all generated Datasets in the form of: {'train': Dataset, 'valid':Dataset, 'test':Dataset}.
        dataloader_train (pytorch dataloader object): Training [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloader_valid (pytorch dataloader object): Validation [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloader_test (pytorch dataloader object): Testing [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloaders (dict): Dictionary of all generated Dataloaders in the form of: {'train': Dataloader, 'valid':Dataloader, 'test':Dataloader}.
        class_weights (tensor): Values of class weights, for imbalanced datasets, to be used to weight loss functions. See [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
        sampler_weights (tensor): Values of vlass weights, for imbalanced datasets, to be used to sample from the dataset using Pytroch WeightedRandomSampler. Affects only training dataset not validation or testing. See [https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

    """

    def __init__(self,folder,name=None,label_table=None,instance_id=False,add_extension=False,out_channels=1,WW=None,WL=None,path_col="path",label_col="label",extension="dcm",transforms=None,random_state=100,sample=1.0,split=False,ignore_zero_img=False,normalize=True,batch_size=16,shuffle=True,weighted_sampler=False,num_workers=0,): #OK

        set_random_seed(random_state)

        self.root = path_fix(folder)
        self.extension = extension

        self.out_channels = out_channels
        self.WL = WL
        self.WW = WW

        self.ignore_zero_img = ignore_zero_img

        self.path_col = path_col
        self.label_col = label_col

        self.split = split
        self.sample = sample

        self.transform = transforms
        if self.transform == None:
            self.transform = {k:None for k in ['train', 'valid', 'test']}
        self.random_state = random_state

        self.normalize = normalize

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Assign name for the dataset object
        if not name: self.name = current_time(human=False) + "_dataset_"
        else: self.name = name

        # if label_table != None: Use table if provided
        if isinstance(label_table, str) or isinstance(label_table, pd.DataFrame):
            if isinstance(label_table, str): self.table = pd.read_csv(label_table)

            elif isinstance(label_table, pd.DataFrame): self.table = deepcopy(label_table)

            self.classes = self.table[self.label_col].unique().tolist()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            if instance_id:
                if add_extension:
                    self.table[self.path_col] = self.table.apply(lambda row: self.root + row[self.path_col] + "." + self.extension,axis=1,)
                else:
                    self.table[self.path_col] = self.table.apply(lambda row: self.root + row[self.path_col],axis=1,)

        else:
            # Find classes and create class_to_idx dict from root if table not provided
            self.classes, self.class_to_idx = find_classes(self.root)
            self.table = dicom_images_to_table(self.root, self.extension, self.path_col, self.label_col)
            assert (len(self.table) > 0), "No .{:} files were found in {:}. Please check.".format(self.extension, self.root)

        # Filter out zero images
        if self.ignore_zero_img: self.table = check_zero_image(self.table, self.path_col)

        # Sample from whole dataaset
        self.table = self.table.sample(frac=self.sample, random_state=100)
        self.table.reset_index(drop=True, inplace=True)
        self.table = self._add_uid_column(self.table)

        self.loaders = {}
        self.datasets = {}
        self.tables = {}

        # Split
        if self.split:
            self.valid_percent = self.split["valid"]
            if "test" in self.split:
                self.test_percent = self.split["test"]
                temp_size = self.valid_percent + self.test_percent
                self.train_percent = 1.0 - temp_size
                self.table_train, temp = train_test_split(self.table,test_size=temp_size,stratify=self.table[[self.label_col]],random_state=self.random_state,)
                self.table_valid, self.table_test = train_test_split(temp,test_size=self.test_percent / temp_size, stratify=temp[[self.label_col]],random_state=self.random_state,)
                self.idx_train, self.idx_valid, self.idx_test = (self.table_train.index.tolist(),self.table_valid.index.tolist(),self.table_test.index.tolist(),)
                self.dataset_test = torch.utils.data.Subset(self, self.idx_test)
                self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,)
                self.datasets["test"] = self.dataset_test
                self.loaders["test"] = self.dataloader_test
                self.tables["test"] = self.table_test

            else:
                self.train_percent = 1.0 - self.valid_percent
                self.table_train, self.table_valid = train_test_split(self.table,test_size=self.valid_percent,stratify=self.table[[self.label_col]],random_state=self.random_state,)
                self.idx_train, self.idx_valid = (self.table_train.index.tolist(),self.table_valid.index.tolist(),)

            self.dataset_train = torch.utils.data.Subset(self, self.idx_train)
            self.dataset_valid = torch.utils.data.Subset(self, self.idx_valid)
            self.dataloader_valid = torch.utils.data.DataLoader(self.dataset_valid,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,)

            self.datasets["valid"] = self.dataset_valid
            self.loaders["valid"] = self.dataloader_valid
            self.tables["valid"] = self.table_valid
        else:
            self.table_train = self.table
            self.idx_train = self.table_train.index.tolist()
            self.dataset_train = self


        # Determine class weights for imbalanced datasets. This can be used to weight the loss functions.
        self.class_weights = class_weight.compute_class_weight( class_weight="balanced", classes=self.classes, y=self.table_train.loc[:, self.label_col].values,)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        # Code below is adapted and modified from https://github.com/ptrblck/pytorch_misc/blob/master/weighted_sampling.py
        # See this discussion for how to implement WeightedRandomSampler: https://discuss.pytorch.org/t/is-weightedsampler-really-useful/40057/2
        target_train_labels = torch.tensor([self.class_to_idx[x] for x in self.table_train.label.tolist()])
        class_sample_count = torch.tensor(
            [(target_train_labels == t).sum() for t in torch.unique(target_train_labels, sorted=True)])
        weight = 1. / class_sample_count.float()
        self.sampler_weight = torch.tensor([weight[t] for t in target_train_labels])


        if weighted_sampler:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(self.sampler_weight, len(self.sampler_weight))
            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,sampler=sampler,)

        else:
            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,)

        self.datasets["train"] = self.dataset_train
        self.loaders["train"] = self.dataloader_train
        self.tables["train"] = self.table_train

    def __len__(self): #OK
        """
        Returns length in whole dataset.
        """
        return len(self.table)

    def __getitem__(self, idx): #OK
        """
        Defines how an instance/image is processed to be fed into the neural network.
        """
        path = self.table.iloc[idx][self.path_col]
        label = self.table.iloc[idx][self.label_col]
        label_id = self.class_to_idx[label]
        uid = self.table.iloc[idx]["uid"]

        img = ImageObject(path, self.out_channels, self.transform[self._find_subset(idx)], self.WW, self.WL)
        if self.normalize:
            img = self._normalize(img)
        return img, label_id, uid

    def info(self):
        """
        Returns breakdown of different attributes of your dataset.
        """

        info = pd.DataFrame.from_dict(
            ({key: str(value) for key, value in self.__dict__.items()}).items()
        )
        info.columns = ["Property", "Value"]
        for i in ["train", "valid", "test"]:
            try:
                info.loc[len(info.index)] = [
                    i + " dataset size",
                    len(self.tables[i]),
                ]
            except:
                pass
        return info

    def data_stat(self, plot=True, figsize=(8, 6), cmap="viridis"): #OK
        """
        Displays distribution of classes across subsets as table or plot.

        Args:

            plot (bool): True, display data as figure. False, display data as table.
            figsize (tuple): size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        Returns:

            pandas dataframe: if plot=False

        Examples:

        ```python
        ds = radtorch.data.ImageDataset(folder='data/4CLASS/', out_channels=1)
        ds.data_stat()
        ```
        <div style="text-align:center"><img src="../assets/1.png" style="width:50%; height:50%" /></div>

        """
        table_dict = {"train": self.table_train}
        if hasattr(self, "table_valid"):
            table_dict["valid"] = self.table_valid
        if hasattr(self, "table_test"):
            table_dict["test"] = self.table_test

        d, c, i, n = [], [], [], []
        for k, v in table_dict.items():
            for l, j in self.class_to_idx.items():
                d.append(k)
                c.append(l)
                i.append(j)
                n.append(v[self.label_col].value_counts()[l].sum())
        df = pd.DataFrame(list(zip(d, c, i, n)), columns=["Dataset", "Class", "Class_idx", "Count"])
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = sns.barplot(x="Dataset", y="Count", hue="Class", data=df, palette=cmap)
            self._show_values_on_bars(ax)
        else:
            return df

    def view_batch(self, subset="train", figsize=(15, 5), cmap="gray", num_images=False, rows=2): #OK
        """
        Displays a batch from a certain subset.

        Args:

            subset (string): Datasubset to use: either 'train', 'valid', or 'test'.
            figsize (tuple): Size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            num_images (int): Number of displayed image. Usually equals batch_size unless otherwise specified.
            rows (int): Number of rows.

        Returns:

            figure: figure containing samples

        Examples:

        ```python
        ds = radtorch.data.ImageDataset(folder='data/4CLASS/', out_channels=1)
        ds.view_batch()
        ```
        <div style="text-align:center"><img src="../assets/2.png" style="width:80%; height:80%" /></div>


        """
        loader = self.loaders[subset]
        images, labels, uid = next(iter(loader))
        images = images.cpu()
        labels = labels.cpu().numpy()
        batch = images.shape[0]
        if num_images:
            assert (num_images <= batch), "Error: Selected number of images is less than batch size. Displaying a batch instead."
            batch = num_images
        fig = plt.figure(figsize=figsize)
        for i in np.arange(batch):
            ax = fig.add_subplot(rows, int(batch / rows), i + 1, xticks=[], yticks=[])
            if images[i].shape[0] == 3:
                img = images[i].moveaxis(0,-1)
                if self.normalize: ax.imshow(img, cmap=cmap)
                else: ax.imshow(img/255., cmap=cmap)
            elif images[i].shape[0] == 1:
                ax.imshow(images[i].squeeze(), cmap=cmap)
            ax.set_title(self.classes[labels[i]])

    def view_image(self, id=0, figsize=(25, 5), cmap="gray"): #OK

        """
        Displays separate images/channels of an image.

        Args:

            id (int): Target image id in `dataset.table` (row index).
            figsize (tuple): size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        Returns:

            figure: figure containing samples

        Examples:

        ```python
        ds = radtorch.data.ImageDataset(folder='data/4CLASS/', out_channels=3)
        ds.view_image(id=9)
        ```
        <div style="text-align:center"><img src="../assets/3.png" style="width:80%; height:80%" /></div>

        ```python
        ds = radtorch.data.ImageDataset(folder='data/4CLASS/', out_channels=3, WW=[1500, 350, 80], WL=[-600, 50, 40])
        ds.view_image(id=9)
        ```
        <div style="text-align:center"><img src="../assets/4.png" style="width:80%; height:80%" /></div>


        """
        img, label, uid = self[id]
        num_channels = img.shape[0]
        assert num_channels > 1, 'Error: Selected image does not multiple channels. Please check.'
        fig = plt.figure(figsize=figsize)
        channels = [img[i, :, :] for i in range(0, num_channels)]
        for i in range(0, num_channels):
            ax = fig.add_subplot(1, num_channels, i + 1, xticks=[], yticks=[])
            ax.imshow(channels[i], cmap=cmap)
            ax.set_title("channel " + str(i))

    def _normalize(self, x):
        if torch.is_tensor(x):
            x = x - torch.min(x)
            x = x / torch.max(x)
            return x
        else:
            x = x - np.min(x)
            x = x / np.max(x)
            return x

    def _add_uid_column(self, df, length=10):
        df["uid"] = [int(str(uuid.uuid1().int)[:length]) for i in range(len(df))]
        return df

    def _show_values_on_bars(self, axs):
        # https://stackoverflow.com/a/51535326
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    def _find_subset(self, idx):
        if idx in self.idx_train:
            return "train"
        elif idx in self.idx_valid:
            return "valid"
        elif idx in self.idx_test:
            return "test"


class VolumeObject(): #OK
    """
    Creates an Image Volume Object (4D tensor) from series images contained in a folder.

    Parameters:
        directory (str): Folder containing series/sequence images. Images must be DICOM files.
        out_channels (int): Number of output channels. Only 1 and 3 channels supported.
        transforms (list): Albumentations transformations. See [https://albumentations.ai/docs/getting_started/image_augmentation/](https://albumentations.ai/docs/getting_started/image_augmentation/).
        WW (int or list, optional): Window width for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
        WL (int or list, optional): Window level for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).

    Returns:
        tensor: 4D tensor with dimensions = [channels, number_images/depth, width, height]. See [https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)

    Examples:

        ```python
        >>> i = radtorch.data.VolumeObject(directory='data/PROTOTYPE/DIRECTORY/abdomen/abd_1')
        >>> i.shape

        torch.Size([1, 40, 512, 512])

        ```

    """
    def __new__(cls, directory, out_channels=1, transforms=None, WW=None, WL=None, resample_spacing=[-1,-1,-1], resample_slices=None, **kwargs):
        volume_tensor, min, max, orig_spacing  = directory_to_tensor(directory, 'dcm', transforms, out_channels, WW, WL)
        if resample_spacing !=[-1,-1,-1] or resample_slices !=None:
            volume_tensor = resample_dicom_volume(volume_tensor, orig_spacing, resample_spacing, resample_slices)
        return volume_tensor


class VolumeDataset(Dataset): #OK
    """
    Dataset object for DICOM Volume. Creates dataset(s) and dataloader(s) ready for training using radtorch or pytorch directly.

    Parameters:
        folder (str): Parent folder containing images. `radtorch.VolumeDataset` expects files to be arranged in the following structure:
                    ```
                    root/
                        class_1/
                                sequence_1/
                                            image_1
                                            image_2
                                            ...
                                sequence_2/
                                            image_1
                                            image_2
                                            ...
                        class_2/
                                sequence_1/
                                            image_1
                                            image_2
                                            ...
                                sequence_2/
                                            image_1
                                            image_2
                                            ...
                        ...

                    ```
        name (str, optional): Name to be give to the dataset. If none provided, the current date and time will be used to created a generic dataset name. (default=None)
        label_table (str|dataframe, optional): The table containing data labels for your images. Expected table should contain at least 2 columns: image path column and a label column. Table can be string path to CSV or a pandas dataframe object.(default=None)
        use_file (bool): True to use pre-generated/resampled volume files. To use Volume files:

            1. Volume files should be created using `radtorch.data.VolumeObject`

            2. Saved with extension `.pt`.

            3. Placed in the sequence folder.

        extension (str, optional): Type/Extension of images.
        out_channels (int, optional): Number of output channels.
        WW (int or list, optional): Window width for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
        WL (int or list, optional): Window level for DICOM images. Single value if using 1 channel or list of 3 values for 3 channels. See [https://radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
        path_col (str, optional): Name of the column containing image path data in the label_table.
        label_col (str, optional): Name of the column containing label data in the label_table.
        study_col (str, optional): Name of the column containing study id in label_table.
        transforms (dict): Dictionary of Albumentations transformations in the form of {'train': .. , 'valid': .. , 'test': .. }. See https://albumentations.ai/docs/getting_started/image_augmentation/. NOTE: If using already resampled/created volume files, transformation should be applied during volume creation not dataset i.e. Transforms specified here have no effect during training.
        random_state (int, optional): Random seed.
        sample (float, optional): Sample or percent of the overall data to be used.
        split (dict): dictionary defining how data will be split for training/validation/testing. Follows the sturcture {'valid': float, 'test': float} or {'valid':'float'} in case no testing subset is needed. The percent of the training subset is infered automatically.
        normalize (bool, optional): True to normalize image data between 0 and 1.
        batch_size (int, optional): Dataloader batch size.
        shuffle (bool, optional): True to shuffle images during training.
        weighted_sampler (bool, optional): True to use a weighted sampler for unbalanced datasets. See https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler.
        num_workers (int, optional): Dataloader CPU workers.


    Attributes:
        classes (list): List of generated classes/labels.
        class_to_idx (dict): Dictionary of generated classes/labels and corresponding class/label id.
        idx_train (list): List of index values of images/instances used for training subset. These refer to index of `ImageDataset.table`.
        idx_valid (list):List of index values of images/instances used for validation subset. These refer to index of `ImageDataset.table`.
        idx_test (list): List of index values of images/instances used for testing subset. These refer to index of `ImageDataset.table`.
        table (pandas dataframe): Table of images , paths and their labels.
        table_train (pandas dataframe): Table of images used for training. Subset of `ImageDataset.table`.
        table_valid (pandas dataframe): Table of images used for validation. Subset of `ImageDataset.table`.
        table_test (pandas dataframe): Table of images used for testing. Subset of `ImageDataset.table`.
        tables (dict): Dictionary of all generated tables in the form of: {'train': table, 'valid':table, 'test':table}.
        dataset_train (pytorch dataset object): Training [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        dataset_valid (pytorch dataset object): Validation [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        dataset_test (pytorch dataset object): Testing [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
        datasets (dict): Dictionary of all generated Datasets in the form of: {'train': Dataset, 'valid':Dataset, 'test':Dataset}.
        dataloader_train (pytorch dataloader object): Training [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloader_valid (pytorch dataloader object): Validation [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloader_test (pytorch dataloader object): Testing [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloaders (dict): Dictionary of all generated Dataloaders in the form of: {'train': Dataloader, 'valid':Dataloader, 'test':Dataloader}.
        class_weights (tensor): Values of class weights, for imbalanced datasets, to be used to weight loss functions. See [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
        sampler_weights (tensor): Values of vlass weights, for imbalanced datasets, to be used to sample from the dataset using Pytroch WeightedRandomSampler. Affects only training dataset not validation or testing. See [https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

    Examples:

        ```python
        import radtorch
        import albumentations as A

        # Specify image transformations
        T = A.Compose([A.Resize(256,256)])

        # Create dataset object
        ds = radtorch.data.VolumeDataset(
                                        folder='data/PROTOTYPE/DIRECTORY/',
                                        split={'valid':0.3, 'test':0.3},
                                        out_channels=1,
                                        transforms={'train':T,'valid': T,'test': T},
                                         )

        ds.data_stat()
        ```
        <div style="text-align:center"><img src="../assets/7.png" style="width:50%; height:50%" /></div>

        ```python
        ds.table
        ```
        <div style="text-align:center"><img src="../assets/8.png" style="width:50%; height:50%" /></div>


    """

    def __init__(self,folder,name=None,label_table=None,use_file=False,extension='dcm', out_channels=1,WW=None,WL=None,path_col="path",label_col="label",study_col="study_id",transforms=None,random_state=100,sample=1.0,split=False,normalize=True,batch_size=16,shuffle=True,weighted_sampler=False,num_workers=0,): #OK

        set_random_seed(random_state)

        self.root = path_fix(folder)
        self.extension = extension
        self.use_file = use_file

        self.out_channels = out_channels
        self.WL = WL
        self.WW = WW

        self.path_col = path_col
        self.label_col = label_col
        self.study_col = study_col

        self.split = split
        self.sample = sample

        self.transform = transforms
        if self.transform == None:
            self.transform = {k:None for k in ['train', 'valid', 'test']}
        self.random_state = random_state

        self.normalize = normalize

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Assign name for the dataset object
        if not name: self.name = current_time(human=False) + "_dataset_"
        else: self.name = name

        # if label_table != None: Use table if provided
        if isinstance(label_table, str) or isinstance(label_table, pd.DataFrame):
            if isinstance(label_table, str): self.table = pd.read_csv(label_table)

            elif isinstance(label_table, pd.DataFrame): self.table = deepcopy(label_table)

            self.classes = self.table[self.label_col].unique().tolist()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        else:
            # Find classes and create class_to_idx dict from root if table not provided
            self.classes, self.class_to_idx,self.table = dicom_volume_to_table(self.root, self.extension, self.path_col, self.label_col, self.study_col, self.use_file)
            assert (len(self.table) > 0), "No files were found in {:}. Please check.".format(self.root)


        # Sample from whole dataaset
        self.table = self.table.sample(frac=self.sample, random_state=100)
        self.table.reset_index(drop=True, inplace=True)
        self.table = self._add_uid_column(self.table)

        self.loaders = {}
        self.datasets = {}
        self.tables = {}

        # Split
        if self.split:
            self.valid_percent = self.split["valid"]
            if "test" in self.split:
                self.test_percent = self.split["test"]
                temp_size = self.valid_percent + self.test_percent
                self.train_percent = 1.0 - temp_size
                self.table_train, temp = train_test_split(self.table,test_size=temp_size,stratify=self.table[[self.label_col]],random_state=self.random_state,)
                self.table_valid, self.table_test = train_test_split(temp,test_size=self.test_percent / temp_size, stratify=temp[[self.label_col]],random_state=self.random_state,)
                self.idx_train, self.idx_valid, self.idx_test = (self.table_train.index.tolist(),self.table_valid.index.tolist(),self.table_test.index.tolist(),)
                self.dataset_test = torch.utils.data.Subset(self, self.idx_test)
                self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,)
                self.datasets["test"] = self.dataset_test
                self.loaders["test"] = self.dataloader_test
                self.tables["test"] = self.table_test

            else:
                self.train_percent = 1.0 - self.valid_percent
                self.table_train, self.table_valid = train_test_split(self.table,test_size=self.valid_percent,stratify=self.table[[self.label_col]],random_state=self.random_state,)
                self.idx_train, self.idx_valid = (self.table_train.index.tolist(),self.table_valid.index.tolist(),)

            self.dataset_train = torch.utils.data.Subset(self, self.idx_train)
            self.dataset_valid = torch.utils.data.Subset(self, self.idx_valid)
            self.dataloader_valid = torch.utils.data.DataLoader(self.dataset_valid,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,)

            self.datasets["valid"] = self.dataset_valid
            self.loaders["valid"] = self.dataloader_valid
            self.tables["valid"] = self.table_valid
        else:
            self.table_train = self.table
            self.idx_train = self.table_train.index.tolist()
            self.dataset_train = self

        # Determine class weights for imbalanced datasets. This can be used to weight the loss functions.
        self.class_weights = class_weight.compute_class_weight( class_weight="balanced", classes=self.classes, y=self.table_train.loc[:, self.label_col].values,)
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)

        # Create weights for WeightedRandomSampler : https://stackoverflow.com/a/60813495
        class_counts = [r["Count"] for i, r in self.data_stat(plot=False).iterrows() if r["Dataset"] == "train" and r["Class"] in self.classes]
        num_samples = sum(class_counts)
        labels = [self.class_to_idx[i] for i in self.table_train[self.label_col].tolist()]
        class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
        self.sampler_weights = torch.DoubleTensor([class_weights[labels[i]] for i in range(int(num_samples))])

        if weighted_sampler:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(self.sampler_weights, self.batch_size)
            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,sampler=sampler,)

        else:
            self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,)

        self.datasets["train"] = self.dataset_train
        self.loaders["train"] = self.dataloader_train
        self.tables["train"] = self.table_train

    def __len__(self): #OK
        """
        Returns length in whole dataset.
        """
        return len(self.table)

    def __getitem__(self, idx):
        """
        Defines how an instance/image is processed to be fed into the neural network.
        """
        path = self.table.iloc[idx][self.path_col]
        label = self.table.iloc[idx][self.label_col]
        label_id = self.class_to_idx[label]
        uid = self.table.iloc[idx]["uid"]

        if self.use_file:
            img = torch.load(path)
        else:
            img, min, max, orig_spacing = directory_to_tensor(path, self.extension, self.transform[self._find_subset(idx)], self.out_channels, self.WW, self.WL)

        if self.normalize:
            img = self._normalize(img)

        return img, label_id, uid

    def data_stat(self, plot=True, figsize=(8, 6), cmap="viridis"): #OK
        """
        Displays distribution of classes across subsets as table or plot.

        Args:

            plot (bool): True, display data as figure. False, display data as table.
            figsize (tuple): size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        Returns:

            pandas dataframe: if plot=False
        """
        table_dict = {"train": self.table_train}
        if hasattr(self, "table_valid"):
            table_dict["valid"] = self.table_valid
        if hasattr(self, "table_test"):
            table_dict["test"] = self.table_test

        d, c, i, n = [], [], [], []
        for k, v in table_dict.items():
            for l, j in self.class_to_idx.items():
                d.append(k)
                c.append(l)
                i.append(j)
                n.append(v[self.label_col].value_counts()[l].sum())
        df = pd.DataFrame(list(zip(d, c, i, n)), columns=["Dataset", "Class", "Class_idx", "Count"])
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = sns.barplot(x="Dataset", y="Count", hue="Class", data=df, palette=cmap)
            self._show_values_on_bars(ax)
        else:
            return df

    def view_study(self, id, plane='axial', figsize=(15, 15), cols=5, rows=5, start=0, end=-1): #OK
        """
        Show sample images from a study.

        !!! warning "Warning"
            This works only with single channel images. Multiple channels are not supported yet here.

        Args:

            id (int): Target study id in `dataset.table` (row index).
            plane (str): Anatomical plane to display the images in. Options: 'axial', 'coronal' or 'sagittal'.
            figsize (tuple): Size of the displayed figure.
            cols (int): Number of columns.
            rows (int): Number of rows.
            start (int): id of the first image to display.
            end (int): id of the last image to display.

        Returns:

            figure: figure containing images from study.

        Examples:

            ```python
            import radtorch
            import albumentations as A

            # Specify image transformations
            T = A.Compose([A.Resize(256,256)])

            # Create dataset object
            ds = radtorch.data.VolumeDataset(
                                            folder='data/PROTOTYPE/DIRECTORY/',
                                            split={'valid':0.3, 'test':0.3},
                                            out_channels=1,
                                            transforms={'train':T,'valid': T,'test': T},
                                 )
            ```
            ```python
            ds.view_study(id=0, plane='axial')
            ```
            <div style="text-align:center"><img src="../assets/9.png" style="width:50%; height:50%" /></div>


            ```python
            ds.view_study(id=0, plane='coronal', start=150)
            ```
            <div style="text-align:center"><img src="../assets/10.png" style="width:50%; height:50%" /></div>


        """
        path = self.table.iloc[id][self.path_col]
        if self.use_file:
            img = torch.load(path)
        else:
            img, min, max, orig_spacing = directory_to_tensor(path, self.extension, self.transform[self._find_subset(id)], self.out_channels, self.WW, self.WL)
        if self.normalize:
            img = self._normalize(img)

        fig = plt.figure(figsize=figsize)

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                         axes_pad=0.0,
                         )
        if img.shape[0] == 1:
            img = img.squeeze(0) #DHW
            if plane == 'coronal':
                img = img.moveaxis(0, 1) #HDW
            if plane == 'sagittal': # > WDH
                img = img.moveaxis(-1,0)  #

            for ax, im in zip(grid, img[start:end, :, :]):
                ax.grid(False)
                ax.axis('off')
                ax.imshow(im.squeeze(), cmap='gray')
        else:
            img = torch.moveaxis(img, 0,-1)
            for ax, im in zip(grid, img[start:end, :, :]):
                ax.grid(False)
                ax.axis('off')
                ax.imshow(im, cmap='gray')
        plt.show()

    def _normalize(self, x):
        if torch.is_tensor(x):
            x = x - torch.min(x)
            x = x / torch.max(x)
            return x
        else:
            x = x - np.min(x)
            x = x / np.max(x)
            return x

    def _add_uid_column(self, df, length=10):
        df["uid"] = [int(str(uuid.uuid1().int)[:length]) for i in range(len(df))]
        return df

    def _show_values_on_bars(self, axs):
        # https://stackoverflow.com/a/51535326
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    def _find_subset(self, idx):
        if idx in self.idx_train:
            return "train"
        elif idx in self.idx_valid:
            return "valid"
        elif idx in self.idx_test:
            return "test"
