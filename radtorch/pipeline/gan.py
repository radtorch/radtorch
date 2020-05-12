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

    def __init__(self,
               data_directory,
               generator_noise_size,
               generator_num_features,
               generator_output_image_size,
               discriminator_num_features,
               discriminator_input_image_size,
               generator_noise_type='normal',
               table=None,
               image_path_column='IMAGE_PATH'
               image_label_column='IMAGE_LABEL',
               is_path=True,
               is_dicom=False,
               mode='RAW',
               wl=None,
               batch_size=16,
               num_workers=0,
               label_smooth=False,
               discriminator='dcgan',
               generator='dcgan',
               epochs=10,
               discrinimator_optimizer='Adam',
               generator_optimizer='Adam',
               discrinimator_optimizer_param={},
               generator_optimizer_param={},
               generator_learning_rate=0.0001,
               discriminator_learning_rate=0.0001,
               generator_output_image_channels=3,
               discriminator_input_image_channels=3,
               sampling=1.0,
               transformations='default',
               beta1=0.5,
               beta2=0.999):

    self.data_directory=data_directory
    self.is_dicom=is_dicom
    self.table=table
    self.image_path_column=image_path_column
    self.image_label_column=image_label_column
    self.is_path=is_path
    selfl.num_workers=num_workers
    self.sampling=sampling
    self.mode=mode
    self.wl=wl

    self.d=discriminator
    self.g=generator
    self.g_output_image_size=generator_output_image_size
    self.g_output_image_channels=generator_output_image_channels
    self.d_input_image_size=discriminator_input_image_size
    self.d_input_image_channels=discriminator_input_image_channels
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
    self.label_smooth=lable_smooth
    self.beta1=beta1
    self.beta2=beta2

    if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    transforms.ToTensor()])
        else:
            self.transformations=transforms.Compose([
                transforms.Resize((self.d_input_image_size, self.d_input_image_size)),
                transforms.ToTensor()])

    # >> need to create dataset/dataloader here
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

    if self.d='dcgan':
      self.D=DCGAN_Discriminator(num_input_channels=self.d_input_image_channels,num_discriminator_features=self.d_num_features, input_image_size=self.d_input_image_size,  kernel_size=4)

    if self.g='dcgan':
      self.G=DCGAN_Generator(noise_size=self.noise_size, num_generator_features=self.g_num_features, num_output_channels=self.g_output_image_channels, target_image_size=self.g_output_image_size)

    self.D = self.D.to(self.device)
    self.G = self.G.to(self.device)

    self.D_optimizer=self.nn_optimizer(type=self.d_optimizer, model=self.D, learning_rate=self.d_learning_rate, [self.beta1, self.beta2])
    self.G_optimizer=self.nn_optimizer(type=self.g_optimizer, model=self.G, learning_rate=self.g_learning_rate, [self.beta1, self.beta2])



    def run(self, num_generated_images=16, show_images=True, figure_size=(10,10)):

        self.training_metrics=[]

        self.generated_samples=[]

        for epochs in range tqdm(range(self.epochs))):

            epoch_start=time.time()

            for batch_number, (images, labels, paths) in enumerate(self.dataloader):

                batch_start=time.time()

                d_loss, d_real_loss, d_fake_loss = self.train_discriminator(generator=self.G,
                                                                  discriminator=self.D,
                                                                  train_images=images,
                                                                  discriminator_optimizer=self.D_optimizer,
                                                                  input_noise_size=self.g_noise_size,
                                                                  label_smooth=self.label_smooth,
                                                                  noise_type=self.g_noise_type)

                g_loss = self.train_generator(generator=self.G,
                               discriminator=self.D,
                               train_images=images,
                               generator_optimizer=self.G_optimizer,
                               input_noise_size=self.g_noise_size,
                               label_smooth=self.label_smooth
                               noise_type=self.g_noise_type)

                training_metrics.append((d_loss.item(),  g_loss.item(), d_real_loss.item(), d_fake_loss.item()))

                batch_end=time.time()

                log("Epoch : {:03d}/{} : [D_loss: {:.4f}, d_loss_real_images {:.4f}%, d_loss_fake_images {:.4f}] [G_loss: {:.4f}%] [Time: {:.4f}s]".format(epoch, self.epochs, d_loss.item(), d_real_loss.item(), d_fake_loss.item(), g_loss.item(), batch_end-batch_start))


            self.G.eval()
            generated_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=num_generated_images)
            generated_noise.to(self.device)
            sample = self.G(generated_noise)
            generated_samples.append(sample)
            if show_images:
                plot_images(generated_samples, titles=[], figure_size=figure_size)
            self.G.train()

        self.trained_G = self.G
        self.trained_D = self.D
        return self.trained_D, self.trained_G, self.training_metrics, self.samples


    def generate_noise(self, noise_size, noise_type, num_images=25):
      if noise_type =='normal': generated_noise = np.random.uniform(-1, 1, size=(num_images, noise_size))
      elif noise_type == 'gaussian':generated_noise = np.random.normal(0, 1, size=(num_images, noise_size))
      else:
        # log('Noise type not specified/recognized. Please check.')
        pass
      generated_noise = torch.from_numpy(generated_noise).float()
      return generated_noise


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
      log('Optimizer selected is '+type)
      return optimizer


    def real_loss(self, D_out, smooth=False):
        batch_size = D_out.size(0)
        # label smoothing
        if smooth: labels = torch.ones(batch_size)*0.9 # smooth, real labels = 0.9
        else: labels = torch.ones(batch_size) # real labels = 1
        # move labels to GPU if available
        labels=labels.to(self.device)
        # binary cross entropy with logits loss
        criterion = nn.BCEWithLogitsLoss()
        # calculate loss
        loss = criterion(D_out.squeeze(), labels)
        return loss


    def train_generator(self, generator, discriminator, train_images, generator_optimizer, input_noise_size, label_smooth, noise_type):
        '''
        Define steps to train the generator network
        Input parameters:
            generator = Generator Neural Network
            discriminator = Discriminator Neural Network
            train_images = Images within current training batch
            generator_optimizer = current state of optimizer of the generator network
            input_noise_size = size of the noise to be generated
            noise_type = Noise type (Default=None)
        Output:
            g_loss = loss of generator network
        Training Steps:
            1. Obtain size of training real images batch tensor (batch_tensor_dim)
            2. Generate fake images with same size (fake_images)
            3. Calculate generator network loss on fake images with flipped lables(g_loss)
            4. Back propagation (.backward)
            5. Update optimizer (optimizer.step)
        '''

        #1. Obtain size of training real images batch tensor (batch_tensor_dim)
        batch_tensor_dim = train_images.size(0)

        #2. Generate fake images with same size (fake_images)
        z = self.generate_noise(noise_size=input_noise_size, noise_type=noise_type, num_images=batch_tensor_dim)
        z = z.to(self.device)
        generator_optimizer.zero_grad()
        fake_images = generator(z)
        D_fake = discriminator(fake_images)

        #3. Calculate generator network loss on fake images with flipped lables(g_loss)
        g_loss = self.real_loss(D_fake, smooth=label_smooth) # use real loss to flip labels

        #4. Back propagation (.backward)
        g_loss.backward()

        #5. Update optimizer (optimizer.step)
        generator_optimizer.step()
        return g_loss


    def train_discriminator(self, generator, discriminator, train_images, discriminator_optimizer, input_noise_size, label_smooth, noise_type):
        '''
        Define steps to train the discriminator network
        Input parameters:
            generator = Generator Neural Network
            discriminator = Discriminator Neural Network
            train_images = Images within current training batch
            discriminator_optimizer = current state of optimizer of the discriminator network
            input_noise_size = size of the noise to be generated
            noise_type = Noise type (Default=None)
        Output:
            d_loss = discriminator loss total
            d_real_loss = discriminator loss on real images
            d_fake_loss = discriminator loss on fake images
        Training Steps:
            1. Calculate discriminator network loss on real images (d_real_loss)
            2. Obtain size of training real images batch tensor (batch_tensor_dim)
            3. General fake images with same size (fake_images)
            4. Calculate discriminator network loss on fake images (d_fake_loss)
            5. Sum both losses into one total (d_loss)
            6. Back propagation (.backward)
            7. Update optimizer (optimizer.step)
        '''
        #1. Calculate discriminator network loss on real images (d_real_loss)
        discriminator_optimizer.zero_grad()
        D_real = discriminator(train_images)
        d_real_loss = self.real_loss(D_real, smooth=label_smooth) #Smooth noise applied by default to D_loss on real images

        #2. Obtain size of training real images batch tensor (batch_tensor_dim)
        batch_tensor_dim = train_images.size(0)

        #3. General fake images with same size (fake_images)
        z = self.generate_noise(noise_size=input_noise_size, noise_type=noise_type, num_images=batch_tensor_dim)
        z = z.to(self.device)
        fake_images = generator(z)

        #4. Calculate discriminator network loss on fake images (d_fake_loss)
        D_fake = discriminator(fake_images)
        d_fake_loss = models.fake_loss(D_fake)

        #5. Sum both losses into one total (d_loss)
        d_loss = d_real_loss + d_fake_loss

        #6. Back propagation (.backward)
        d_loss.backward()

        #7. Update optimizer (optimizer.step)
        discriminator_optimizer.step()
        return d_loss, d_real_loss, d_fake_loss
