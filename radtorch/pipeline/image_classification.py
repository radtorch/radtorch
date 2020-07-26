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

## Code Last Updated: 08/01/2020

from ..settings import *
from ..core import *
from ..utils import *
# from ..beta import *


class Image_Classification():
    
    def __init__(
                self,
                data_directory,
                name=None,
                is_dicom=False,
                table=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                is_path=True,
                mode='RAW',
                wl=None,
                balance_class=False,
                balance_class_method='upsample',
                interaction_terms=False,
                normalize=((0,0,0), (1,1,1)),
                batch_size=16,
                num_workers=0,
                sampling=1.0,
                test_percent=0.2,
                valid_percent=0.2,
                custom_resize=False,
                model_arch='resnet50',
                pre_trained=True,
                unfreeze=False,
                type='nn_classifier',
                cv=True,
                stratified=True,
                num_splits=5,
                parameters={},
                learning_rate=0.0001,
                epochs=10,
                optimizer='Adam',
                loss_function='CrossEntropyLoss',
                lr_scheduler=None,
                custom_nn_classifier=None,
                loss_function_parameters={},
                optimizer_parameters={},
                transformations='default',
                extra_transformations=None,
                device='auto',
                auto_save=False,
                **kwargs):

        self.data_directory=data_directory
        self.is_dicom=is_dicom
        self.table=table
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.mode=mode
        self.wl=wl
        self.balance_class=balance_class
        self.balance_class_method=balance_class_method
        self.interaction_terms=interaction_terms
        self.normalize=normalize
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.sampling=sampling
        self.test_percent=test_percent
        self.valid_percent=valid_percent
        self.custom_resize=custom_resize
        self.model_arch=model_arch
        self.pre_trained=pre_trained
        self.unfreeze=unfreeze
        self.type=type
        self.cv=cv
        self.stratified=stratified
        self.num_splits=num_splits
        self.parameters=parameters
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.lr_scheduler=lr_scheduler
        self.custom_nn_classifier=custom_nn_classifier
        self.loss_function_parameters=loss_function_parameters
        self.optimizer_parameters=optimizer_parameters
        self.transformations=transformations
        self.extra_transformations=extra_transformations
        self.device=device
        self.name=name

        if self.name==None:
            self.name = 'image_classification_'+datetime.now().strftime("%m%d%Y%H%M%S")+'.pipeline'

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.type not in SUPPORTED_CLASSIFIER:
            log('Error! Classifier type not supported.')
            pass
        if 'data_processor' not in self.__dict__.keys(): self.data_processor=Data_Processor(**self.__dict__)
        if 'feature_extractor' not in self.__dict__.keys(): self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.master_dataloader, **self.__dict__)
        if 'extracted_feature_dictionary' not in self.__dict__.keys():
            self.train_feature_extractor=Feature_Extractor(dataloader=self.data_processor.train_dataloader, **self.__dict__)
            self.test_feature_extractor=Feature_Extractor(dataloader=self.data_processor.test_dataloader, **self.__dict__)
        # if auto_safe:
        #     global log_dir
        #     outfile=open(log_dir+self.name,'wb')
        #     pickle.dump(self,outfile)
        #     outfile.close()

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, gui=False):
        log('Starting Image Classification Pipeline', gui=gui)
        set_random_seed(100)
        if self.type!='nn_classifier':
            log('Phase 1: Feature Extraction.', gui=gui)

            if 'extracted_feature_dictionary' in self.__dict__.keys():
                log('Features Already Extracted. Loading Previously Extracted Features', gui=gui)
            else:
                log('Extracting Training Features', gui=gui)
                self.train_feature_extractor.run(gui=gui)
                log('Extracting Testing Features', gui=gui)
                self.test_feature_extractor.run(gui=gui)
                self.extracted_feature_dictionary={
                                                    'train':{'features':self.train_feature_extractor.features, 'labels':self.train_feature_extractor.labels_idx, 'features_names': self.train_feature_extractor.feature_names,},
                                                    'test':{'features':self.test_feature_extractor.features, 'labels':self.test_feature_extractor.labels_idx, 'features_names': self.test_feature_extractor.feature_names,}
                                                    }

            log('Phase 2: Classifier Training.', gui=gui)
            log ('Running Classifier Training.', gui=gui)
            self.classifier=Classifier(**self.__dict__, )
            self.classifier.run(gui=gui)
            self.trained_model=self.classifier
            self.train_metrics=self.classifier.train_metrics
            # self.feature_selector=Feature_Selector(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
            log ('Classifier Training completed successfully.', gui=gui)

        else:
            self.classifier=NN_Classifier(**self.__dict__)
            self.trained_model, self.train_metrics=self.classifier.run()
            log ('Classifier Training completed successfully.', gui=gui)

    def metrics(self, figure_size=(700,350)):
        return show_metrics([self.classifier],  figure_size=figure_size)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log ('Pipeline exported successfully to '+output_path)
        except:
            log ('Error! Pipeline could not be exported.')
            pass

    def cam(self, target_image_path, target_layer, type='scorecam', figure_size=(10,5), cmap='jet', alpha=0.5):

        if type =='cam':
            wrapped_model = CAM(model=self.classifier.trained_model.to(self.device), target_layer=target_layer, device=self.device)
        elif type == 'gradcam':
            wrapped_model = GradCAM(model=self.classifier.trained_model.to(self.device), target_layer=target_layer, device=self.device)
        elif type == 'gradcampp':
            wrapped_model = GradCAMpp(model=self.classifier.trained_model.to(self.device), target_layer=target_layer, device=self.device)
        elif type == 'smoothgradcampp':
            wrapped_model = SmoothGradCAMpp(model=self.classifier.trained_model.to(self.device), target_layer=target_layer, device=self.device)
        elif type == 'scorecam':
            wrapped_model = ScoreCAM(model=self.classifier.trained_model.to(self.device), target_layer=target_layer,  device=self.device)

        if self.is_dicom:
            image=dicom_to_narray(target_image_path, self.mode, self.wl)
            image=Image.fromarray(image)
        else:
            image=Image.open(target_image_path).convert('RGB')

        prep_img=self.data_processor.transformations(image)
        prep_img=prep_img.unsqueeze(0)
        prep_img = prep_img.to(self.device)
        cam, idx = wrapped_model(prep_img)
        _, _, H, W = prep_img.shape
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=True)

        output_image=prep_img.squeeze(0).squeeze(0).cpu().numpy()
        output_image=np.moveaxis(output_image, 0, -1)

        plt.figure(figsize=figure_size)

        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.gca().set_title('Target Image')
        plt.imshow(output_image, cmap=plt.cm.gray)

        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.gca().set_title(type.upper())
        plt.imshow(cam.squeeze().cpu().numpy(), cmap=cmap, alpha=1)

        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.gca().set_title('OVERLAY')
        plt.imshow(output_image, cmap=plt.cm.gray)
        plt.imshow(cam.squeeze().cpu().numpy(), cmap=cmap, alpha=alpha)

        plt.show()

    def deploy(self, title="Image Classification", colab=False):
        if os.path.exists("/ui_framework.py"):
            os.remove("/ui_framework.py")
        else:
            file_operation=open('/ui_framework.py', 'a')
            file_operation.write(ui_framework)
            file_operation.close()
        export_model_name='/'+self.name+'.saved_model'
        self.export(export_model_name)
        if colab:
            colab_streamlit_crossover()
        subprocess.call(['streamlit', 'run', '/ui_framework.py',  'image_classification', export_model_name, title])
