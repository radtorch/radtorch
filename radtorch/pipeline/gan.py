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
               image_path_column='IMAGE_PATH',
               image_label_column='IMAGE_LABEL',
               is_path=True,
               is_dicom=False,
               mode='RAW',
               wl=None,
               batch_size=16,
               normalize=((0,0,0),(1,1,1)),
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
               beta2=0.999,
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
        self.label_smooth=label_smooth
        self.beta1=beta1
        self.beta2=beta2

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

        if self.g=='dcgan':
          self.G=DCGAN_Generator(noise_size=self.g_noise_size, num_generator_features=self.g_num_features, num_output_channels=self.g_output_image_channels, target_image_size=self.g_output_image_size)

        self.G.apply(self.weights_init)
        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        self.D_optimizer=self.nn_optimizer(type=self.d_optimizer, model=self.D, learning_rate=self.d_learning_rate, **{'betas':(self.beta1, self.beta2)})
        self.G_optimizer=self.nn_optimizer(type=self.g_optimizer, model=self.G, learning_rate=self.g_learning_rate, **{'betas':(self.beta1, self.beta2)})

        self.fixed_noise = torch.randn(self.batch_size, self.g_noise_size, 1, 1, device=self.device)

    def run(self, num_generated_images=16, show_images=True, figure_size=(10,10)):

        real_label=1
        fake_label=0

        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        self.train_metrics=[]

        self.generated_samples=[]

        for epoch in tqdm(range(self.epochs)):

            epoch_start=time.time()

            for batch_number, (images, labels, paths) in enumerate(self.dataloader):

                batch_start=time.time()

                # (1) Train D
                ###########################
                ## Train with all-real batch
                self.D.zero_grad()
                # Format batch
                images = images.to(self.device)
                b_size = images.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                # Forward pass real batch through D
                output = self.D(images).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                # D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                              # generated_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=b_size)
                generated_noise=torch.randn((b_size,self.g_noise_size, 1, 1), device=self.device)
                # Generate fake image batch with G
                fake = self.G(generated_noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                # D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.D_optimizer.step()

                ############################
                # (2) Train G
                ###########################
                self.G.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.D(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                # D_G_z2 = output.mean().item()
                # Update G
                self.G_optimizer.step()

                self.train_metrics.append([errD.item(),  errG.item(), errD_real.item(), errD_fake.item()])

                batch_end=time.time()

                log("Epoch : {:03d}/{} : [D_loss: {:.4f}, d_loss_real_images {:.4f}, d_loss_fake_images {:.4f}] [G_loss: {:.4f}] [Time: {:.4f}s]".format(epoch, self.epochs, errD.item(), errD_real.item(), errD_fake.item(), errG.item(), batch_end-batch_start))


            self.G.eval()
            # generated_noise = self.generate_noise(noise_size=self.g_noise_size, noise_type=self.g_noise_type, num_images=num_generated_images)
            sample = self.G(self.fixed_noise)
            sample = sample.cpu().detach().numpy()
            sample = [np.moveaxis(x, 0, -1) for x in sample]
            self.generated_samples.append(sample)
            if show_images:
                plot_images(sample , titles=None, figure_size=figure_size)
            self.G.train()

        self.trained_G = self.G
        self.trained_D = self.D
        self.train_metrics=pd.DataFrame(data=self.train_metrics, columns = ['D_loss', 'd_loss_real_images', 'd_loss_fake_images', 'G_loss'])


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
      # log('Optimizer selected is '+type)
      return optimizer


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # def real_loss(self, D_out, smooth=False):
    #   D_out=D_out.to(self.device)
    #   batch_size = D_out.size(0)
    #   # label smoothing
    #   if smooth: labels = torch.ones(batch_size)*0.9 # smooth, real labels = 0.9
    #   else: labels = torch.ones(batch_size) # real labels = 1
    #   # move labels to GPU if available
    #   labels=labels.to(self.device)
    #   # binary cross entropy with logits loss
    #   criterion = nn.BCEWithLogitsLoss()
    #   # calculate loss
    #   loss = criterion(D_out.squeeze(), labels)
    #   return loss


    # def generate_noise(self, noise_size, noise_type, num_images=25):
    #   if noise_type =='normal': generated_noise = np.random.uniform(-1, 1, size=(num_images, noise_size))
    #   elif noise_type == 'gaussian':generated_noise = np.random.normal(0, 1, size=(num_images, noise_size))
    #   else:
    #     # log('Noise type not specified/recognized. Please check.')
    #     pass
    #   generated_noise = torch.from_numpy(generated_noise).float()
    #   generated_noise=generated_noise.to(self.device)
    #   return generated_noise

    def metrics(self, figure_size=(700,350)):
      return show_metrics([self],  figure_size=figure_size)
