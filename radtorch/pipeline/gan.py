# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

from ..settings import *
from ..core import *
from ..utils import *



class GAN():

    """

    Description
    -----------
    Generative Advarsarial Networks Pipeline.


    Parameters
    ----------

    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - batch_size (integer, optional): Batch size for dataloader. defult=16.

    - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

    - normalize (bolean/False or Tuple, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=((0,0,0),(1,1,1)).

    - label_smooth (boolean, optioanl): by default, labels for real images as assigned to 1. If label smoothing is set to True, lables of real images will be assigned to 0.9. default=True. (Source: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels)

    - epochs (integer, required): training epochs. default=10.

    - generator (string, required): type of generator network. Options = {'dcgan', 'vanilla'}. default='dcgan'

    - discriminator (string, required): type of discriminator network. Options = {'dcgan', 'vanilla'}. default='dcgan'

    - image_channels (integer, required): number of output channels for discriminator input and generator output. default=1

    - generator_noise_type (string, optional): shape of noise to sample from. Options={'normal', 'gaussian'}. default='normal'. (https://github.com/soumith/ganhacks#3-use-a-spherical-z)

    - generator_noise_size (integer, required): size of the noise sample to be generated. default=100

    - generator_num_features (integer, required): number of features/convolutions for generator network. default=64

    - image_size (integer, required): iamge size for discriminator input and generator output.default=128

    - discriminator_num_features (integer, required): number of features/convolutions for discriminator network.default=64

    - generator_optimizer (string, required): generator network optimizer type. If set to 'auto', pipeline assigns appropriate optimizer automatically. Please see radtorch.settings for list of approved optimizers. default='auto'.

    - generator_optimizer_param (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation. default={'betas':(0.5,0.999)} for Adam optimizer.

    - discrinimator_optimizer (string, required): discrinimator network optimizer type. If set to 'auto', pipeline assigns appropriate optimizer automatically. Please see radtorch.settings for list of approved optimizers. default='auto'.

    - discrinimator_optimizer_param (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation. default={'betas':(0.5,0.999)} for Adam optimizer.

    - generator_learning_rate (float, required): generator network learning rate. default=0.0001.

    - discriminator_learning_rate (float, required): discrinimator network learning rate. default=0.0001.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    - loss (string, optional): type of loss to be applied. Options{'minmax', 'wasserstein'}. default='minmax'

    - num_critics (integer, required with wgan): number of critics/disciminator to train before training generator. default=2

    Methods
    -------

    .run(self, verbose='batch', show_images=True, figure_size=(10,10))

        - Runs the GAN training.

        - Parameters:

            - verbose (string, required): amount of data output. Options {'batch': display info after each batch, 'epoch': display info after each epoch}.default='batch'

            - show_images (boolean, optional): True to show sample of generatot generated images after each epoch.

            - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10)

     .sample(figure_size=(10,10), show_labels=True)

        - Displays a sample of real data.

        - Parameters:

            - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10).

            - show_labels (boolean, optional): show labels on top of images. default=True.

    .info()

        - Displays different parameters of the generative adversarial network.

    .metrics(figure_size=(700,350))

        - Displays training metrics for the GAN.

        - Explanation of metrics:

            - D_loss: Total loss of discriminator network on both real and fake images.

            - G_loss: Loss of discriminator network on detecting fake images as real.

            _ d_loss_real: Loss of discriminator network on detecting real images as real.

            - d_loss_fake: Loss of discriminator network on detecting fake images as fake.

        - Parameters:

            - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(700,350).

    .generate(noise_size=100, figure_size=(5,5))

    """

    def __init__(self,
               data_directory,
               generator_noise_size=100,
               generator_num_features=64,
               image_size=128,
               discriminator_num_features=64,
               generator_noise_type='normal',
               table=None,
               image_path_column='IMAGE_PATH',
               image_label_column='IMAGE_LABEL',
               is_path=True,
               is_dicom=False,
               mode='RAW',
               wl=None,
               batch_size=16,
               normalize=((0,0,0),(1,1,1)),
               num_workers=0,
               label_smooth=True,
               discriminator='dcgan',
               generator='dcgan',
               epochs=10,
               discrinimator_optimizer='auto',
               generator_optimizer='auto',
               discrinimator_optimizer_param={'betas':(0.5,0.999)},
               generator_optimizer_param={'betas':(0.5,0.999)},
               generator_learning_rate=0.0001,
               discriminator_learning_rate=0.0001,
               image_channels=1,
               sampling=1.0,
               transformations='default',
               num_critics=2,
               device='auto'):

        self.data_directory=data_directory
        self.is_dicom=is_dicom
        self.table=table
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.num_workers=num_workers
        self.sampling=sampling
        self.mode=mode
        self.wl=wl
        self.device=device
        self.transformations=transformations
        self.batch_size=batch_size
        self.normalize=normalize

        self.d=discriminator
        self.g=generator
        self.g_output_image_size=image_size
        self.g_output_image_channels=image_channels
        self.d_input_image_size=image_size
        self.d_input_image_channels=image_channels
        self.g_noise_size=generator_noise_size
        self.g_noise_type=generator_noise_type
        self.d_num_features=discriminator_num_features
        self.g_num_features=generator_num_features

        self.g_learning_rate=generator_learning_rate
        self.d_learning_rate=discriminator_learning_rate
        self.d_optimizer=discrinimator_optimizer
        self.g_optimizer=generator_optimizer
        self.d_optimizer_param=discrinimator_optimizer_param
        self.g_optimizer_param=generator_optimizer_param
        self.epochs=epochs
        self.label_smooth=label_smooth
        self.num_critics=num_critics

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion=nn.BCELoss()

        if isinstance(self.table, str):
            if self.table!='':
                self.table=pd.read_csv(self.table)
        elif isinstance(self.table, pd.DataFrame):
            self.table=self.table
        else: self.table=create_data_table(directory=self.data_directory, is_dicom=self.is_dicom, image_path_column=self.image_path_column, image_label_column=self.image_label_column)


        if self.transformations=='default':
            if self.is_dicom:
                self.transformations=transforms.Compose([
                        transforms.Resize((self.d_input_image_size, self.d_input_image_size)),
                        transforms.transforms.Grayscale(self.d_input_image_channels),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])])
            elif self.d_input_image_channels != 3:
                self.transformations=transforms.Compose([
                    transforms.Resize((self.d_input_image_size, self.d_input_image_size)),
                    transforms.transforms.Grayscale(self.d_input_image_channels),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])])
            else:
                self.transformations=transforms.Compose([
                    transforms.Resize((self.d_input_image_size, self.d_input_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])])

        self.dataset=RADTorch_Dataset(
                                        data_directory=self.data_directory,
                                        table=self.table,
                                        is_dicom=self.is_dicom,
                                        mode=self.mode,
                                        wl=self.wl,
                                        image_path_column=self.image_path_column,
                                        image_label_column=self.image_label_column,
                                        is_path=self.is_path,
                                        sampling=self.sampling,
                                        transformations=self.transformations)

        self.dataloader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        if self.d=='dcgan':
          self.D=DCGAN_Discriminator(num_input_channels=self.d_input_image_channels,num_discriminator_features=self.d_num_features, input_image_size=self.d_input_image_size,  kernel_size=4)
        elif self.d=='vanilla':
          self.D=GAN_Discriminator(input_image_size=self.d_input_image_size, intput_num_channels=self.d_input_image_channels, device=self.device)
        elif self.d=='wgan':
          self.D=WGAN_Discriminator(num_input_channels=self.d_input_image_channels,num_discriminator_features=self.d_num_features, input_image_size=self.d_input_image_size,  kernel_size=4)


        if self.g=='dcgan':
          self.G=DCGAN_Generator(noise_size=self.g_noise_size, num_generator_features=self.g_num_features, num_output_channels=self.g_output_image_channels, target_image_size=self.g_output_image_size)
        elif self.g=='vanilla':
          self.G=GAN_Generator(noise_size=self.g_noise_size, target_image_size=self.g_output_image_size, output_num_channels=self.g_output_image_channels, device=self.device)
        elif self.g=='wgan':
          self.G=WGAN_Generator(noise_size=self.g_noise_size, num_generator_features=self.g_num_features, num_output_channels=self.g_output_image_channels, target_image_size=self.g_output_image_size)

        self.G.apply(self.weights_init)
        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        if self.d_optimizer=='auto':
            if self.d in ['dcgan', 'vanilla']:
                self.D_optimizer=self.nn_optimizer(type='Adam', model=self.D, learning_rate=self.d_learning_rate, **self.d_optimizer_param)
            elif self.d =='wgan':
                self.D_optimizer=self.nn_optimizer(type='RMSprop', model=self.D, learning_rate=self.d_learning_rate)
        else:
            self.D_optimizer=self.nn_optimizer(type=self.d_optimizer, model=self.D, learning_rate=self.d_learning_rate, **self.d_optimizer_param)

        if self.g_optimizer=='auto':
            if self.g in ['dcgan', 'vanilla']:
                self.G_optimizer=self.nn_optimizer(type='Adam', model=self.G, learning_rate=self.d_learning_rate, **self.d_optimizer_param)
            elif self.g =='wgan':
                self.G_optimizer=self.nn_optimizer(type='RMSprop', model=self.G, learning_rate=self.d_learning_rate)
        else:
            self.G_optimizer=self.nn_optimizer(type=self.g_optimizer, model=self.G, learning_rate=self.g_learning_rate, **self.g_optimizer_param)

        # self.fixed_noise = torch.randn(self.batch_size, self.g_noise_size, 1, 1, device=self.device)
        self.fixed_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=self.batch_size)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def sample(self, figure_size=(10,10), show_labels=True):
        show_dataloader_sample(self.dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name = False,)

    def nn_optimizer(self, type, model, learning_rate, **kw):
      if type not in supported_nn_optimizers:
          log('Error! Optimizer not supported yet. Please check radtorch.settings.supported_nn_optimizers')
          pass
      elif type=='Adam':
          optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate, **kw)
      elif type=='AdamW':
          optimizer=torch.optim.AdamW(params=model.parameters(), lr=learning_rate, **kw)
      elif type=='SparseAdam':
          optimizer=torch.optim.SparseAdam(params=model.parameters(), lr=learning_rate, **kw)
      elif type=='Adamax':
          optimizer=torch.optim.Adamax(params=model.parameters(), lr=learning_rate, **kw)
      elif type=='ASGD':
          optimizer=torch.optim.ASGD(params=model.parameters(), lr=learning_rate, **kw)
      elif type=='RMSprop':
          optimizer=torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, **kw)
      elif type=='SGD':
          optimizer=torch.optim.SGD(params=model.parameters(), lr=learning_rate, **kw)
      return optimizer

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def generate_noise(self, noise_size, noise_type, num_images=25):
        if noise_type =='normal': generated_noise = np.random.uniform(-1, 1, size=(num_images, noise_size))
        elif noise_type == 'gaussian':generated_noise = np.random.normal(0, 1, size=(num_images, noise_size))
        else:
            log('Noise type not specified/recognized. Please check.')
            pass
        generated_noise=torch.from_numpy(generated_noise).float()
        generated_noise=generated_noise.to(self.device)
        generated_noise=generated_noise.unsqueeze(-1)
        generated_noise=generated_noise.unsqueeze(-1)
        return generated_noise

    def run(self, verbose='batch', show_images=True, figure_size=(10,10)):
        if self.label_smooth:
            real_label=0.9

        else:
            real_label=1.0
        fake_label=0.0

        if self.d=='wgan':
            real_label = -1
            fake_label = 1

        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        self.train_metrics=[]

        self.generated_samples=[]

        num_batches=len(self.dataloader)

        for epoch in tqdm(range(self.epochs)):

            epoch_errD=[]
            epoch_errG=[]
            epoch_d_loss_real=[]
            epoch_d_loss_fake=[]

            epoch_start=time.time()

            for batch_number, (images, labels, paths) in enumerate(self.dataloader):

                batch_start=time.time()


                if self.d in ['vanilla', 'dcgan']:

                    self.D.zero_grad()
                    images = images.to(self.device)
                    b_size = images.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                    output = self.D(images).view(-1)
                    errD_real = self.criterion(output, label)
                    errD_real.backward()

                    generated_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=b_size)

                    fake = self.G(generated_noise)
                    label.fill_(fake_label)
                    output = self.D(fake.detach()).view(-1)
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    self.D_optimizer.step()

                    self.G.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = self.D(fake).view(-1)
                    errG = self.criterion(output, label)
                    errG.backward()
                    self.G_optimizer.step()


                elif self.d =='wgan':

                    self.D.zero_grad()
                    images = images.to(self.device)
                    b_size = images.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                    output1 = self.D(images).view(-1)
                    errD_real = self.criterion(output1, label)

                    generated_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=b_size)
                    fake = self.G(generated_noise)
                    label.fill_(fake_label)
                    output2 = self.D(fake.detach()).view(-1)
                    errD_fake = self.criterion(output1, label)
                    errD = torch.mean(output2)-torch.mean(output1)
                    errD.backward()
                    self.D_optimizer.step()

                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    if batch_number % self.num_critics == 0:
                        self.G.zero_grad()
                        label.fill_(real_label)
                        output = self.D(fake).view(-1)
                        errG=-torch.mean(output)
                        errG.backward()
                        self.G_optimizer.step()



                self.train_metrics.append([errD.item(),  errG.item(), errD_real.item(), errD_fake.item()])
                epoch_d_loss_real=[errD_real.item()]
                epoch_d_loss_fake=[errD_fake.item()]
                epoch_errD=[errD.item()]
                epoch_errG=[errG.item()]


                batch_end=time.time()
                if verbose=='batch':
                    log("[Epoch:{:03d}/{:03d}, Batch{:03d}/{:03d}] : [D_loss: {:.4f}, G_loss: {:.4f}] [d_loss_real {:.4f}, d_loss_fake {:.4f}] [Time: {:.4f}s]".format(epoch, self.epochs, batch_number, num_batches, errD.item(),errG.item(), errD_real.item(), errD_fake.item(), batch_end-batch_start))

            epoch_end=time.time()
            if verbose=='epoch':
                log("[Epoch:{:03d}/{:03d}] : [D_loss: {:.4f}, G_loss: {:.4f}] [d_loss_real {:.4f}, d_loss_fake {:.4f}] [Time: {:.4f}s]".format(epoch, self.epochs, mean(epoch_errD), mean(epoch_errG), mean(epoch_d_loss_real), mean(epoch_d_loss_fake), epoch_end-epoch_start))

            self.G.eval()
            sample = self.G(self.fixed_noise)
            sample = sample.cpu().detach().numpy()
            sample = [np.moveaxis(x, 0, -1) for x in sample]
            self.generated_samples.append(sample)
            if show_images:
                plot_images(sample , titles=None, figure_size=figure_size)
            self.G.train()

        self.trained_G = self.G
        self.trained_D = self.D
        self.train_metrics=pd.DataFrame(data=self.train_metrics, columns = ['D_loss', 'G_loss', 'd_loss_real_images', 'd_loss_fake_images', ])

    def metrics(self, figure_size=(700,350)):
      return show_metrics([self],  figure_size=figure_size, type='GAN')

    def generate(self, noise_type='normal', figure_size=(4,4)):
        generated_noise = generate_noise(noise_size=self.noise_size, noise_type=noise_type, num_images=1)
        generated_image = self.trained_G(generated_noise).detach().cpu()
        generated_image = generated_image.data.cpu().numpy()
        fig = plt.figure(figsize=figure_size)
        implot = plt.imshow(generated_image[-1][-1], cmap='gray')

    def export_generated_images(self, output_folder, figure_size=(10,10), zip=False):#<<<<<<<<<<<<<<<<<<<< NEEDS FIX
        for images in self.generated_samples:
            cols = int(math.sqrt(len(images)))
            n_images = len(images)
            titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
            fig = plt.figure(figsize=figure_size)
            for n, (image, title) in enumerate(zip(images, titles)):
                a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
                if image.shape[2]==1:
                    image=np.squeeze(image, axis=2)
                    plt.gray()
                image_max = np.amax(image)
                image_min = np.amin(image)
                if image_max >255 or image_min <0 :
                  image=np.clip(image, 0, 255)
                plt.axis('off')
            plt.axis('off')
            plt.savefig(fname=output_folder+str(self.generated_samples.index(images))+'.png', transparent=True)
        if zip:
            os.system("zip -r output_folder/generated_images.zip output_folder")
