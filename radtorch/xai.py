
from .utils import *
from .inference import *

import shap
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad

class XAI():
    def __init__(self, classifier, use_best=True, device='auto'):
        self.classifier = classifier
        self.dataset = classifier.dataset
        if use_best:
            self.model = classifier.best_model
        else:
            self.model = classifier.model
        self.device = select_device(device)
        if 'cpu' in self.device.type:
            self.use_cuda = False
        else:
            self.use_cuda = True

    def _view_cam_sidebyside(self, img_array, cam_array, cam_type, figsize=(15,5), alpha=0.5, cmap='jet', prediction=False):
        y = int(img_array.shape[0]/20)
        if len(img_array.shape) > 2:
            img_array = np.moveaxis(img_array, 0, -1)/255.
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.text(0, y, 'Original Image', color='white', backgroundcolor='black' )
        plt.imshow(img_array, cmap=plt.cm.gray)
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.text(0, y, 'Class Activation Map ({:})'.format(cam_type), color='white', backgroundcolor='black' )
        plt.imshow(cam_array, cmap=cmap, alpha=1)
        plt.subplot(1, 3, 3)
        plt.axis('off')
        if prediction:
            plt.text(0, y, prediction, color='white', backgroundcolor='black')
        plt.imshow(img_array, cmap=plt.cm.gray)
        plt.imshow(cam_array, cmap=cmap, alpha=0.5)
        plt.show()

    def CAM(self, img_path, target_layers, type='fullgrad', plot=True, export=False, prediction=True, figsize=(15,5), alpha=0.5, cmap='jet',):
        input_tensor = image_to_tensor(img_path=img_path, transforms=self.dataset.transform['valid'], extension=self.dataset.extension, out_channels=self.dataset.out_channels, WW=self.dataset.WW, WL = self.dataset.WL).to(self.device)
        assert type in ['gradcam', 'xgradcam', 'scorecam', 'gradcamplusplus', 'ablation', 'eigen', 'eigengrad', 'layercam', 'fullgrad'], 'CAM type selected is not in supported list.'
        if type == 'gradcam': wrapped_model = GradCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'xgradcam': wrapped_model = XGradCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type  == 'scorecam': wrapped_model = ScoreCAM(model=self.model, target_layers=[None], use_cuda=self.use_cuda)
        elif type == 'gradcamplusplus': wrapped_model = GradCAMPlusPlus(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'ablation': wrapped_model = AblationCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'eigen': wrapped_model = EigenCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'eigengrad': wrapped_model = EigenGradCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'layercam': wrapped_model = LayerCAM(model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)
        elif type == 'fullgrad': wrapped_model = FullGrad(model=self.model, target_layers=[], use_cuda=self.use_cuda)
        map = wrapped_model(input_tensor)
        if export:
            return map.squeeze()
        else:
            if prediction:
                I = Inference(classifier=self.classifier, use_best_model=True)
                pred = I.predict(img_path=img_path, display_image=False, human=False)[0][0]
                pred = ('class: {:4}, prob: {:.2f}%'.format(pred['class'], pred['prob']*100))
                self._view_cam_sidebyside(input_tensor.squeeze().cpu().numpy(), map.squeeze(), cam_type=type, prediction=pred, figsize=figsize, alpha=alpha, cmap=cmap)
            else:
                self._view_cam_sidebyside(input_tensor.squeeze().cpu().numpy(), map.squeeze(), cam_type=type, figsize=figsize, alpha=alpha, cmap=cmap)

    def SHAP(self, subset_1='train', subset_1_samples=10, subset_2='test', subset_2_samples=5):
        batch = next(iter(self.classifier.dataset.loaders[subset_1]))
        images, labels, uid = batch
        background = images[:subset_1_samples]
        background = background.to(self.device)
        batch = next(iter(self.classifier.dataset.loaders[subset_2]))
        images, labels, uid = batch
        test_images = images[:subset_2_samples]
        test_images = test_images.to(self.device)

        e = shap.DeepExplainer(self.model.to(self.device), background.float())
        shap_values = e.shap_values(test_images.float())

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        shap.image_plot(shap_numpy, test_numpy)
