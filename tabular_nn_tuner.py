import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from src.nn_tabular_models.node_model import NODEModel
from src.abalone_dataset import AbaloneDataset
from src.training_loop import training_loop
from src.styles import TXT_ACC, TXT_RESET


PROJECT = 'PGs04e04'



class TabularNNTuner:

    def __init__(self, train_data, target, cv_idx):
        
        self.train_data = train_data
        self.target = target

        self.cv_idx = cv_idx

        self.model_dict = {'NODE': NODEModel,}


    def log_training(wandb_run, epoch, model, loss_valid, log_name):
        name = f'{wandb_run.id}_{log_name}'
        path = 'trained_models/' + name + '.pth'
        torch.save({'best_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_loss': loss_valid}, 
                path)
                        
        artifact = wandb.Artifact(name=name, type=log_name)
        artifact.add_file(path)            
        wandb_run.log_artifact(artifact)

    @staticmethod
    def _count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    @staticmethod
    def score(model_class, model_params, dataloader_train, dataloader_val, learning_rate, device, verbose=False):
        model = model_class(**model_params).to(device)

        if verbose:
            print(model_params)
            print('Number of trainable parameters in model: ', TabularNNTuner._count_parameters(model))

        best_loss, best_epoch, best_model_state = training_loop(
                            dataloader_train,
                            dataloader_val,
                            model,
                            optim.Adam(model.parameters(), lr=learning_rate),
                            nn.MSELoss(),
                            num_epochs=1000,
                            device=device, 
                            early_stopping_patience=30,
                            verbose=verbose)
        
        if verbose:
            print('Best loss     ', np.sqrt(best_loss))
            print('Best epochs   ', best_epoch)
        
        return best_loss, best_epoch, best_model_state
    


    def _score_cv(self, model_class, model_params, batch_size, learning_rate, device, verbose=False):
        best_scores = []
        best_epochs = []
        for fold, (idx_train, idx_val) in enumerate(self.cv_idx):
            dataset_train = AbaloneDataset(self.train_data[idx_train], self.target[idx_train])
            dataset_val = AbaloneDataset(self.train_data[idx_val], self.target[idx_val])

            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
            
            if verbose:
                print(f'{TXT_ACC} Fold {fold} {TXT_RESET}')

            best_loss, best_epoch, _ = self.score(model_class, model_params, dataloader_train, dataloader_val, learning_rate, device, verbose)
           
            best_scores.append(np.sqrt(best_loss))
            best_epochs.append(best_epoch)

        return best_scores, best_epochs




    def _objective_cv(self):

        wandb.init(project=PROJECT)
        
        config = wandb.config

        model_class = self.model_dict.get(config['model'])
        model_parameters = {key[6:]: val for key, val in config.items() if key.startswith('model_')}
        scores, epochs = self._score_cv(model_class, 
                                        model_parameters, 
                                        batch_size=config['batch_size'],
                                        learning_rate=config['learning_rate'],
                                        device=torch.device(config['device']) )

        wandb.log({f'score_{i}': sc for i, sc in enumerate(scores)})
        wandb.log({f'epoch_{i}': sc for i, sc in enumerate(epochs)})
        wandb.log({'score_mean': scores.mean()})    



    def tune_parameters(self, model_name):

        config = {
            'name': f'sweep_{model_name}',
            'method': 'bayes',
            'metric': {'goal': 'minimize', 'name': 'score_mean'},
            'parameters': {
                'model':            {'value': model_name},
                'learning_rate':    {'min': 1e-5,  'max': 0.2},
                'batch_size':       {'value':  1024},
                'device':           {'value':  'cuda'},
                # model parameters
                **{'model_'+key: val for key, val in self.model_dict.get(model_name).get_params_grid().items()},
                }
            }

        try:
            sweep_id = wandb.sweep(sweep=config, project=PROJECT)            
            wandb.agent(sweep_id, function=self._objective_cv, count=2)        
        except:
            print('Something went wrong')
        finally:
            wandb.finish()