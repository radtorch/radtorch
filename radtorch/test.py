from radtorch.settings import *

def find_lr(model, train_dataloader, optimizer, device):
    set_random_seed(100)
    training_losses=[]
    learning_rates=[]
    model=model.to(device)
    for i, (inputs, labels, image_paths) in enumerate(train_dataloader)
        model.train()
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item() * inputs.size(0)
        training_losses.append(train_loss)
        learning_rates.append(optimizer.learning_rate)


    return training_losses, learning_rates
