# Models Module <small> radtorch.modelsutils </small>


!!! bug " Documentation Outdated. Please check again later for update."

## create_model
    modelsutils.create_model(model_arch, output_classes, mode,
                pre_trained=True, unfreeze_weights=True)

!!! quote ""
      Creates a PyTorch training neural network model with specified network architecture. Input channels and output classes can be specified.

      **Arguments**

      - model_arch: _(str)_ The architecture of the model neural network. Examples include 'vgg16', 'resnet50', and 'resnet152'.
      - pre_trained: _(boolen)_ Load the pretrained weights of the neural network. (default=True)
      - unfreeze_weights: _(boolen)_ Unfreeze model weights for training.(default=True)
      - output_classes: _(int)_ Number of output classes for image classification problems.
      - mode: _(str)_ 'train' for training model. 'feature_extraction' for feature extraction model

      **Output**

      - Output: _(PyTorch neural network object)_

      **Example**

          my_model = modelsutils.create_model(model_arch='resnet50', output_classes=2)





## create_loss_function
    modelsutils.create_loss_function(type)

!!! quote ""
      Creates a PyTorch training loss function object.

    **Arguments**

    - type: _(str)_  type of the loss functions required.

    **Output**

    - Output: _(PyTorch loss function object)_

    **Example**

        loss = modelsutils.create_loss_function(type='NLLLoss')





##create_optimizer
    modelsutils.create_optimizer(traning_model, optimizer_type, learning_rate)

!!! quote ""

      Creates a PyTorch optimizer object.

    **Arguments**   

    - training_model: _(pytorch Model object)_ target training model.

    - optimizer_type: _(str)_ type of optimizer e.g.'Adam' or 'SGD'.

    - learning_rate: _(float)_ learning rate.

    **Output**

    - Output: _(PyTorch optimizer object)_




## train_model
    train_model(model, train_data_loader, valid_data_loader,
                train_data_set, valid_data_set, loss_criterion,
                optimizer, epochs, device, verbose)

!!! quote ""
      Training loop for pytorch model object.


    **Arguments**   

      - model: _(PyTorch neural network object)_ Model to be trained.

      - train_data_loader: _(PyTorch dataloader object)_ training data dataloader.

      - valid_data_loader: _(PyTorch dataloader object)_ validation data dataloader.

      - train_data_loader: _(PyTorch dataset object)_ training data dataset.

      - valid_data_loader: _(PyTorch dataset object)_ validation data dataset.

      - loss_criterion: _(PyTorch nn object)_ Loss function to be used during training.

      - optimizer: _(PyTorch optimizer object)_ Optimizer to be used during training.

      - epochs: _(int)_ training epochs.

      - device: _(str)_ device to be used for training. This can be 'cpu' or 'cuda'.

      - verbose: _(boolen)_ True to display training messages.

    **Output**

      - model: _(PyTorch neural network object)_ trained model.

      - train_metrics: _(list)_ list of np arrays of training loss and accuracy.

    **Example**

            trained_model, training_metrics = modelsutils.train_model(model=my_model,
                train_data_loader=train_dl, valid_data_loader=valid_dl,
                train_data_set=train_ds, valid_data_set=valid_ds,
                loss_criterion=my_loss, optimizer=my_optim,
                epochs=100, device='cuda', verbose=True)



##model_inference
    modelsutils.model_inference(model, input_image_path,
      inference_transformations=transforms.Compose([transforms.ToTensor()]))

!!! quote ""
      Performs Inference/Predictions on a target image using a trained model.

      **Arguments**

      - model: _(PyTorch Model)_ Trained neural network.

      - input_image_path: _(str)_ path to target image

      - inference_transformations: _(pytorch transforms list)_ pytroch transforms to be performed on the dataset.

      **Output**

      - Output: _(tupe)_ tuple of prediction class id and prediction accuracy percentage.



##supported
      modelsutils.supported()

!!! quote ""

    Returns a list of the currently supported network architectures and loss functions.
