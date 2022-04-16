import torch
from .general import message, set_random_seed, current_time
from copy import deepcopy
from .data import save_checkpoint
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torchinfo import summary



def train_neural_network(model, dataloader, optimizer, criterion, scheduler, device, random_seed):
    #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    set_random_seed(random_seed)
    running_loss = 0.0
    running_correct = 0
    model.train()
    for i, (images, labels, uid) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images.float())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler:
            for s in scheduler:
                if s.__class__.__name__ in ['OneCycleLR', 'CyclicLR']:
                    s.step()

        _, preds = torch.max(outputs.data, 1)
        running_loss += loss.item()*images.size(0)
        running_correct += torch.sum(preds == labels.data)
        # train_acc += (preds == labels).sum().item()

    # epoch_loss = train_loss/len(dataloader) # average of train loss per epoch divided by number of samples in train dataset
    # epoch_acc = 100. * (train_acc / len(dataloader))
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100*(running_correct.double() / len(dataloader.dataset))
    return epoch_loss, epoch_acc


def validate_neural_network(model, dataloader, optimizer, criterion, device, random_seed):
    #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    set_random_seed(random_seed)
    running_loss = 0.0
    running_correct = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels, uid) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            # valid_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            # valid_acc += (preds == labels).sum().item()
            running_loss += loss.item()*images.size(0)
            running_correct += torch.sum(preds == labels.data)
    # epoch_loss = valid_loss/ len(dataloader.dataset)
    # epoch_acc = 100. * (valid_acc / len(dataloader))
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100*(running_correct.double() / len(dataloader.dataset))
    return epoch_loss, epoch_acc


def fit_neural_network(classifier, epochs=20, valid=True, print_every= 1, target_valid_loss='lowest', auto_save_ckpt=False, verbose=2):

    if hasattr (classifier, 'current_epoch'):
        start_epoch=classifier.current_epoch+1
        message(" Resuming training starting at epoch "+ str(start_epoch)+" on "+str(classifier.device))
        classifier.optimizer.load_state_dict(classifier.checkpoint['optimizer_state_dict'])
        message(" Optimizer state loaded successfully.")
    else:
        message(" Starting model training on "+str(classifier.device))
        start_epoch=0
        classifier.train_losses, classifier.valid_losses = [], []
        classifier.train_acc, classifier.valid_acc = [], []
        classifier.valid_loss_min = np.Inf

        if target_valid_loss == 'lowest':
            classifier.target_valid_loss = np.Inf
        else:
            classifier.target_valid_loss = target_valid_loss

        message(" Target Validation loss set to "+ str(classifier.target_valid_loss))


    classifier.original_model = deepcopy(classifier.model)
    classifier.model = classifier.model.to(classifier.device)

    for e in tqdm(range(start_epoch,epochs), desc='Traninig Model: '):

        epoch_train_loss, epoch_train_acc = train_neural_network(classifier.model, classifier.dataloader_train, classifier.optimizer, classifier.criterion, classifier.scheduler,classifier.device, classifier.random_seed)
        classifier.train_losses.append(epoch_train_loss)
        classifier.train_acc.append(epoch_train_acc)

        if valid:
            epoch_valid_loss, epoch_valid_acc = validate_neural_network(classifier.model, classifier.dataloader_valid, classifier.optimizer, classifier.criterion, classifier.device, classifier.random_seed)
            classifier.valid_losses.append(epoch_valid_loss)
            classifier.valid_acc.append(epoch_valid_acc)

            if epoch_valid_loss < classifier.valid_loss_min:
                classifier.valid_loss_min = epoch_valid_loss
                classifier.best_model = deepcopy(classifier.model)
                if epoch_valid_loss <= classifier.target_valid_loss:
                    if auto_save_ckpt:
                        save_checkpoint(classifier=classifier, epochs=epochs, current_epoch=e)
                        save_ckpt, v_loss_dec, v_loss_below_target = True, True, True
                    else:
                        save_ckpt, v_loss_dec, v_loss_below_target = False, True, True
                else:
                    save_ckpt, v_loss_dec, v_loss_below_target = False, True, False
            else:
                save_ckpt, v_loss_dec, v_loss_below_target = False, False, False

            if e % print_every == 0:
                if verbose == 3:
                    message (
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)+
                            " v_loss: {:.5f} (best: {:.5f}) |".format(epoch_valid_loss,classifier.valid_loss_min)+
                            " v_loss dec: {:5} |".format(str(v_loss_dec))+
                            " v_loss below target: {:5} |".format(str(v_loss_below_target))+
                            " ckpt saved: {:5} ".format(str(save_ckpt))
                            )
                elif verbose == 2:
                    message (
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)+
                            " v_loss: {:.5f} (best: {:.5f}) |".format(epoch_valid_loss,classifier.valid_loss_min)+
                            " t_acc: {:.5f} |".format(epoch_train_acc)+
                            " v_acc: {:.5f} |".format(epoch_valid_acc)
                            )
                elif verbose == 1:
                    message (
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)+
                            " v_loss: {:.5f} (best: {:.5f}) |".format(epoch_valid_loss,classifier.valid_loss_min)
                            )

        else:
            if e % print_every == 0:
                if verbose != 0:
                    message(
                            " epoch: {:4}/{:4} |".format(e, epochs)+
                            " t_loss: {:.5f} |".format(epoch_train_loss)
                            )

        metrics_dict = {'train_loss':epoch_train_loss, 'train_accuracy':epoch_train_acc, 'valid_loss':epoch_valid_loss, 'valid_accuracy':epoch_valid_acc}

        if classifier.scheduler != [None]:
            for s in classifier.scheduler:
                if s.__class__.__name__ in ['OneCycleLR', 'CyclicLR']:
                    pass
                elif s.__class__.__name__ =='ReduceLROnPlateau':
                    s.step(metrics_dict[classifier.scheduler_metric])
                else:
                    s.step()
        # print (e, classifier.optimizer.param_groups[0]['lr'])

        if e+1 == epochs:
            message(' Training Finished Successfully!')

    if classifier.valid_loss_min > classifier.target_valid_loss:
        message(current_time()+" CAUTION: Achieved minimum validation loss "+str(classifier.valid_loss_min), " is not less than the set target loss of "+str(classifier.target_valid_loss), gui)

    classifier.train_logs=pd.DataFrame({"train": classifier.train_losses, "valid" : classifier.valid_losses})


def pass_image_via_nn(tensor, model, device, output='logits', top_k=1):
    '''
    Runs image/images through model. The expected image(s) tensor shape is (B, C, W, H). If only 1 image to be passed, then B=1.
    output:
            logits: return logits by last layer of model per each image
            softmax: returns logits passed via softmax layer per each image
            topk: return list of predicted index/label and prediction percent per each image as per top_k specified
    '''
    model = model.to(device)
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor).cpu().detach()
        if output == 'logits':
            return out
        else:
            m = nn.Softmax(dim=1)
            predictions = m(out)
            if output == 'softmax':
                return predictions
            elif output == 'topk':
                out = []
                for i in predictions:
                    pred = torch.topk(i,k=top_k)
                    out.append([pred.indices.numpy().tolist(), pred.values.numpy().tolist()])
                return out


def pass_loader_via_nn(loader, model, device, output='logits', top_k=1, table=False):
    '''
    Same as pass_image_via_nn but for whole loader.
    table: in case of top_k =1, user can export a table with true labels, pred, perc and uid for each instance in loader.
    '''
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        label_list = []
        uid_list = []
        pred_list = []
        perc_list = []
        for i, (imgs, labels, uid) in enumerate(loader):
            label_list = label_list+labels.tolist()
            uid_list = uid_list+uid.tolist()
            imgs = imgs.to(device)
            out = model(imgs.float()).cpu().detach()
            if output == 'logits':
                return out
            else:
                m = nn.Softmax(dim=1)
                predictions = m(out)
                if output == 'softmax':
                    return predictions

                elif output == 'topk':
                    out =  []
                    for i in predictions:
                        pred = torch.topk(i,k=top_k)
                        out.append([pred.indices.numpy().tolist(), pred.values.numpy().tolist()])
                        if top_k == 1:
                            pred_list.append(pred.indices.item())
                            perc_list.append(pred.values.item())
                    if table==True:
                        return pd.DataFrame(list(zip(uid_list, label_list,pred_list, perc_list)),columns =['uid','label_id', 'pred_id', 'percent'])
                    else:
                        return out


def model_details(model, list=False, batch_size=1, channels=3, img_dim=224):
    if isinstance(model, str): model = eval('models.'+model+ "()")
    if list:
        return list(model.named_children())
    else:
        return summary(model, input_size=(batch_size, channels, img_dim, img_dim), depth=channels, col_names=["input_size", "output_size", "num_params"],)
