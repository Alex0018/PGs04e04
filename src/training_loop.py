import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb
import gc

import copy

import torch.nn.functional as F

def train_epoch(dataloader, model, optimizer, loss_criterion, device):
    
    model.train()
    loss = 0.0

    for batch in dataloader:

        data = batch[0].to(device=device)
        target = batch[1].to(device=device)
        
        output = model(data)

        cur_loss = loss_criterion(output, target)
        loss += cur_loss.item()

        cur_loss.backward()
        optimizer.step()        
        optimizer.zero_grad()

    return loss / len(dataloader)


def validate_epoch(dataloader, model, loss_criterion, device):
    model.eval()
    loss = 0.0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            output = model(data)

            preds.extend(output.squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())

            # cur_loss = loss_criterion(output, target)
            # loss += cur_loss.item()

    return loss_criterion(torch.Tensor(preds), torch.Tensor(targets)).item()    
    # return loss / len(dataloader)
 


def training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs, 
                  device,
                  early_stopping_patience,
                  verbose=False):
    
    losses_train = np.zeros(num_epochs)
    losses_valid = np.zeros(num_epochs)
    
    best_loss = np.inf
    best_epoch = -1
    best_model_state = None

    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=verbose)
    
    for epoch in range(num_epochs):          
        loss_train = train_epoch(dataloader_train, model, optimizer, loss_criterion, device)
        loss_valid = validate_epoch(dataloader_valid, model, loss_criterion, device)

        losses_train[epoch] = loss_train
        losses_valid[epoch] = loss_valid        
       
        if loss_valid < (best_loss - 0.00001):
            best_epoch = epoch
            best_loss = loss_valid
            best_model_state = copy.deepcopy(model.state_dict())
            # NB: deepcopy is necessary, otherwise best_model_state will hold a reference 
            # to a current model's state, not a snapshot of the best state as intended
            patience_counter = 0
            if verbose: print(f'* Epoch {str(epoch).rjust(4)}:     loss train {np.sqrt(loss_train):.5f},     loss valid {np.sqrt(loss_valid):.5f}')
        else:
            patience_counter += 1
            if verbose: print(f'  Epoch {str(epoch).rjust(4)}:     loss train {np.sqrt(loss_train):.5f},     loss valid {np.sqrt(loss_valid):.5f}')
            if patience_counter >= early_stopping_patience:
                if verbose: print('Early stopping')
                break    

        scheduler.step(loss_valid)
        

    return best_loss, best_epoch, best_model_state
