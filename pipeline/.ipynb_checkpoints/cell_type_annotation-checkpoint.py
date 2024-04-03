'''
In the context of computer programming and operating systems, a process is an instance of a computer program that is being executed. It contains the program code and its current activity. Each process has a separate memory address space, which means that processes do not inherently share memory with each other. This isolation ensures that processes can run independently without directly interfering with each other's operations.

Main Process
The "main process" typically refers to the primary process under which a program starts running. In the context of data loading for machine learning models, the main process would be the one where your model's training loop runs. This includes computations for forward passes through the model, backward passes for gradient computation, and weight updates during training. The main process is responsible for orchestrating various tasks, including initiating data loading through worker subprocesses if configured to do so.

Subprocesses
"Subprocesses" are additional processes that are spawned by the main process to perform specific tasks. In the context of data loading with multiple workers, each worker would run in its subprocess. These subprocesses can load and preprocess data in parallel to the main process. Because subprocesses have their own memory space, they do not share local variables directly with the main process or with each other. Instead, communication between the main process and its subprocesses is done through inter-process communication mechanisms provided by the operating system, such as pipes or shared memory spaces managed by the framework being used (e.g., PyTorch).

Why Use Subprocesses for Data Loading?
Using subprocesses for data loading in machine learning has several benefits:

Parallelism: Subprocesses can load and preprocess data in parallel to the computation happening in the main process. This ensures that data is ready for the next iteration of the model training loop, minimizing the time the GPU or CPU spends idle waiting for data.
Efficiency: By offloading I/O-bound (e.g., reading files from disk) and CPU-bound (e.g., image resizing, normalization) tasks to subprocesses, the main process can focus exclusively on computation-heavy tasks like executing the model's forward and backward passes.
Scalability: Leveraging multiple CPU cores by using subprocesses can significantly speed up data preprocessing, making it possible to train models faster on large datasets.
Considerations
While using subprocesses for data loading can improve training efficiency, it's important to manage resources wisely. Each subprocess consumes additional memory, and there's overhead involved in managing the communication between the main process and subprocesses. The optimal number of worker subprocesses depends on the specific workload, the data pipeline's complexity, and the hardware capabilities (e.g., the number of CPU cores and available memory).
'''


import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from ..utils.eval import downstream_eval, aggregate_eval_results
from ..utils.data import XDict, TranscriptomicDataset
from typing import List, Union
from .experimental import symbol_to_ensembl
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain

CellTypeAnnotationDefaultModelConfig = {  #Scope: drop_node_rate applies to the dataset level by removing entire nodes (and all their features), whereas                                                  #model_dropout operates within the model's architecture by nullifying the activation of individual neurons.
    'drop_node_rate': 0.3,
    'dec_layers': 1,
    'model_dropout': 0.5,
    'mask_node_rate': 0.75,  #During training, this is the rate at which nodes (cells) are masked (i.e., their features are set to zero). This helps the model                               #learn to predict missing data and can improve generalization.
    'mask_feature_rate': 0.25,  #Similar to mask_node_rate, but for individual features (e.g., gene expressions) within each node. Here, 25% of the features                                   #are randomly masked.
    'dec_mod': 'mlp',
    'latent_mod': 'ae',
    'head_type': 'annotation',
    'max_batch_size': 70000,
}

CellTypeAnnotationDefaultPipelineConfig = {
    'es': 200,  #Early stopping criterion. Training stops if the validation metric does not improve for 'es' epochs. Here, training would stop after 200 epochs                 #without improvement.
    'lr': 5e-3,
    'wd': 1e-7,  # Weight decay. This is a regularization term that helps prevent the weights from growing too large, which can be helpful for avoiding                          #overfitting. Here, it is very low at 1e-7.
    'scheduler': 'plat',  #scheduler: The type of learning rate scheduler. "plat" suggests a plateau scheduler, which reduces the learning rate when a metric                             #stops improving.
    'epochs': 100, # 2000,
    'max_eval_batch_size': 100000,
    'hvg': 3000,
    'patience': 25,  #patience: The number of epochs to wait for an improvement in the validation metric before reducing the learning rate (when using a                             #ReduceLROnPlateau scheduler) or stopping the training (as part of early stopping).
    'workers': 0,   #The number of worker processes used for loading data. A value of 0 means that the data loading will run in the main process, potentially                       #leading to slower data processing.
}
def inference(model, dataloader, split, device, batch_size, eval_dict, label_fields=None, order_required=False):
    if order_required and split:
        warnings.warn('When cell order required to be preserved, dataset split will be ignored.')

    with torch.no_grad():
        model.eval()
        epoch_loss = []
        order_list = []
        pred = []
        label = []
        for i, data_dict in enumerate(dataloader):
            if not order_required and split and np.sum(data_dict['split'] == split) == 0:
                continue
            
            idx = torch.arange(data_dict['x_seq'].shape[0])
            if split:
                data_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == split).values).bool()
            else:
                data_dict['loss_mask'] = torch.ones(data_dict['x_seq'].shape[0]).bool()
            if label_fields:
                data_dict['label'] = data_dict[label_fields[0]]
            for j in range(0, len(idx), batch_size):
                if len(idx) - j < batch_size:
                    cur = idx[j:]
                else:
                    cur = idx[j:j + batch_size]
                input_dict = {}
                for k in data_dict:
                    if k =='x_seq':
                        input_dict[k] = data_dict[k].index_select(0, cur).to(device)
                    elif k not in ['protein_list', 'split']:
                        input_dict[k] = data_dict[k][cur].to(device)
                x_dict = XDict(input_dict)
                out_dict, loss = model(x_dict, data_dict['protein_list'])
                if 'label' in input_dict:
                    epoch_loss.append(loss.item())
                    label.append(out_dict['label'])
                if order_required:
                    order_list.append(input_dict['order_list'])
                pred.append(out_dict['pred'])

        pred = torch.cat(pred)
        if order_required:
            order = torch.cat(order_list)
            order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            pred = pred[order]

        if len(epoch_loss) == 0:
            return {'pred': pred}
        else:
            scores = downstream_eval('annotation', pred, torch.cat(label),
                                           **eval_dict)
            return {'pred': pred,
                    'loss': sum(epoch_loss)/len(epoch_loss),
                    'metrics': scores}

class CellTypeAnnotationPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 overwrite_config: dict = CellTypeAnnotationDefaultModelConfig,
                 pretrain_directory: str = './ckpt',
                 pretrain = True,
                 ):
        # self.pretrain = pretrain
        assert 'out_dim' in overwrite_config, '`out_dim` must be provided in `overwrite_config` for initializing a cell type annotation pipeline. '
        super().__init__(pretrain_prefix, overwrite_config, pretrain_directory) #, self.pretrain
        self.eval_dict = {'num_classes': overwrite_config['out_dim']} 
        self.label_encoders = None

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            train_split: str = 'train',
            valid_split: str = 'valid',
            covariate_fields: List[str] = None,
            label_fields: List[str] = None,
            batch_protein_list: dict = None,
            ensembl_auto_conversion: bool = True,
            device: Union[str, torch.device] = 'cpu',
            ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if train_config:
            config.update(train_config)
        self.model.to(device)
        assert not self.fitted, 'Current pipeline is already fitted and does not support continual training. Please initialize a new pipeline.'
        if batch_protein_list is not None:
            raise NotImplementedError('Batch specific protein set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if len(label_fields) != 1:
            raise NotImplementedError(f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        assert (split_field and train_split and valid_split), '`train_split` and `valid_split` must be specified.'
        # adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion)
        print('!!!!!!commented out the line above me IN def fit because this is proteomics data and no need for ensemble.gene!!!!!!!!!')
        print(f'After filtering, {adata.shape[1]} proteins remain.')
        dataset = TranscriptomicDataset(adata, split_field, covariate_fields, label_fields)
        self.label_encoders = dataset.label_encoders
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=config['workers'])
        optim = torch.optim.AdamW([
            {'params': list(self.model.embedder.parameters()), 'lr': config['lr'] * 0.1,
             'weight_decay': 1e-10},
            {'params': list(self.model.encoder.parameters()) + list(self.model.head.parameters()) + list(
                self.model.latent.parameters()), 'lr': config['lr'],
             'weight_decay': config['wd']},
        ])
        if config['scheduler'] == 'plat':
            scheduler = ReduceLROnPlateau(optim, 'min', patience=config['patience'], factor=0.95)
        else:
            scheduler = None

        train_loss = []
        valid_loss = []
        valid_metric = []
        final_epoch = -1
        best_dict = None

        for epoch in tqdm(range(config['epochs'])):
            self.model.train()
            epoch_loss = []
            train_scores = []

            if epoch < 30:
                for param_group in optim.param_groups[1:]:
                    param_group['lr'] = config['lr'] * (epoch + 1) / 30

            for i, data_dict in enumerate(dataloader):
                print('data_dict:', data_dict)
                input_dict = data_dict.copy()
                del input_dict['protein_list'], input_dict['split']
                input_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == train_split).values).bool()
                input_dict['label'] = input_dict[label_fields[0]] # Currently only support annotating one label
                for k in input_dict:
                    input_dict[k] = input_dict[k].to(device)
                x_dict = XDict(input_dict)
                print('x_dict.keys():', x_dict.keys())
                out_dict, loss = self.model(x_dict, data_dict['protein_list']) #, pretrain = self.pretrain
                print('out_dict.keys():', out_dict.keys())
                with torch.no_grad():
                    train_scores.append(
                        downstream_eval('annotation', out_dict['pred'], out_dict['label'], **self.eval_dict))

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optim.step()
                epoch_loss.append(loss.item())

                if config['scheduler'] == 'plat':
                    scheduler.step(loss.item())

            train_loss.append(sum(epoch_loss) / len(epoch_loss))
            train_scores = aggregate_eval_results(train_scores)
            result_dict = inference(self.model, dataloader, valid_split, device,
                                                config['max_eval_batch_size'], self.eval_dict, label_fields)
            valid_scores = result_dict['metrics']
            valid_loss.append(result_dict['loss'])
            valid_metric.append(valid_scores['f1_score'])

            print(f'Epoch {epoch} | Train loss: {train_loss[-1]:.4f} | Valid loss: {valid_loss[-1]:.4f}')
            print(
                f'Train ACC: {train_scores["acc"]:.4f} | Valid ACC: {valid_scores["acc"]:.4f} | '
                f'Train f1: {train_scores["f1_score"]:.4f} | Valid f1: {valid_scores["f1_score"]:.4f} | '
                f'Train pre: {train_scores["precision"]:.4f} | Valid pre: {valid_scores["precision"]:.4f}')

            if max(valid_metric) == valid_metric[-1]:
                best_dict = deepcopy(self.model.state_dict())
                final_epoch = epoch

            if max(valid_metric) != max(valid_metric[-config['es']:]):
                print(f'Early stopped. Best validation performance achieved at epoch {final_epoch}.')
                break

        assert best_dict, 'Best state dict was not stored. Please report this issue on Github.'
        self.model.load_state_dict(best_dict)
        self.fitted = True
        return self

    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_protein_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu',
                ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if inference_config:
            config.update(inference_config)
        self.model.to(device)
        assert self.fitted, 'Cell type annotation pipeline does not support zero shot setting. Please fine-tune the model on downstream datasets before inference.'
        if batch_protein_list is not None:
            raise NotImplementedError('Batch specific protein set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        # adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion)
        print('!!!!!!commented out the line above me IN def predict because this is proteomics data and no need for ensemble.gene!!!!!!!!!')
        print(f'After filtering, {adata.shape[1]} proteins remain.')
        dataset = TranscriptomicDataset(adata, None, covariate_fields, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=config['workers'])
        return inference(self.model, dataloader, None, device,
                  config['max_eval_batch_size'], self.eval_dict, order_required=True)['pred']

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_protein_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu',
              ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if evaluation_config:
            config.update(evaluation_config)
        self.model.to(device)
        assert self.fitted, 'Cell type annotation pipeline does not support zero shot setting. Please fine-tune the model on downstream datasets before inference.'
        if batch_protein_list is not None:
            raise NotImplementedError('Batch specific protein set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if target_split:
            assert split_field, '`split_filed` must be provided when `target_split` is specified.'
        # adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion, )
        print('!!!!!!commented out the line above me IN def score because this is proteomics data and no need for ensemble.gene!!!!!!!!!')
        print(f'After filtering, {adata.shape[1]} proteins remain.')
        dataset = TranscriptomicDataset(adata, split_field, covariate_fields, label_fields, label_encoders=self.label_encoders)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=config['workers'])
        return inference(self.model, dataloader, target_split, device,
                  config['max_eval_batch_size'], self.eval_dict, label_fields)['metrics']