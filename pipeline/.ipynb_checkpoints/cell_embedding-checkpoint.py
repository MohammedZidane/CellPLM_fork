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
from typing import List, Literal, Union
from .experimental import symbol_to_ensembl
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


CellEmbeddingDefaultModelConfig = {  #Scope: drop_node_rate applies to the dataset level by removing entire nodes (and all their features), whereas                                                  #model_dropout operates within the model's architecture by nullifying the activation of individual neurons.
    'drop_node_rate': 0.3,
    'dec_layers': 1,
    'model_dropout': 0.5,
    'mask_node_rate': 0.75,  #During training, this is the rate at which nodes (cells) are masked (i.e., their features are set to zero). This helps the model                               #learn to predict missing data and can improve generalization.
    'mask_feature_rate': 0.25,  #Similar to mask_node_rate, but for individual features (e.g., gene expressions) within each node. Here, 25% of the features                                   #are randomly masked.
    'dec_mod': 'mlp',
    'latent_mod': 'ae',
    'head_type': 'embedding',  #annotation
    'max_batch_size': 70000,
}

CellEmbeddingDefaultPipelineConfig = {
    'es': 200,  #Early stopping criterion. Training stops if the validation metric does not improve for 'es' epochs. Here, training would stop after 200 epochs                 #without improvement.
    'lr': 5e-3,
    'wd': 1e-7,  # Weight decay. This is a regularization term that helps prevent the weights from growing too large, which can be helpful for avoiding                          #overfitting. Here, it is very low at 1e-7.
    'scheduler': 'plat',  #scheduler: The type of learning rate scheduler. "plat" suggests a plateau scheduler, which reduces the learning rate when a metric                             #stops improving.
    'epochs': 500,  #2000
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
                out_dict, loss, logits = model(x_dict, data_dict['protein_list'])
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
            return {'pred': pred, 'logits': logits}
        else:
            scores = downstream_eval('embedding', pred, torch.cat(label), #annotation
                                           **eval_dict)
            return {'pred': pred,
                    'loss': sum(epoch_loss)/len(epoch_loss),
                    'metrics': scores,
                    'logits': logits}


class CellEmbeddingPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 pretrain_directory: str = './ckpt',
                 ):
        super().__init__(pretrain_prefix, {'head_type': 'embedder'}, pretrain_directory)
        self.label_encoders = None

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,  # A field in adata.obs for representing train-test split
            train_split: str = None,  # A specific split where labels can be utilized for training
            valid_split: str = None,  # A specific split where labels can be utilized for validation
            covariate_fields: List[str] = None,  # A list of fields in adata.obs that contain cellular covariates
            label_fields: List[str] = None,  # A list of fields in adata.obs that contain cell labels
            batch_protein_list: dict = None,  # A dictionary that contains batch and protein list pairs
            ensembl_auto_conversion: bool = True,
            # A bool value indicating whether the function automativally convert symbols to ensembl id
            device: Union[str, torch.device] = 'cpu'
            ):
        # raise NotImplementedError('Currently CellPLM only supports zero shot embedding instead of fine-tuning.')
        print('!!!!!commented out the error that currently CellPLM only suports zero shot embedding because I am trying to make fine tuning on proteomics data!!!!!!!!')

        config = CellEmbeddingDefaultPipelineConfig.copy()
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
        print('!!!!!!commented out the line above me because this is proteomics data and no need for ensemble.gene!!!!!!!!!')
        print(f'After filtering, {adata.shape[1]} proteins remain.')
        # print('label_fields before TranscriptomicsDataset:', label_fields)
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
                input_dict = data_dict.copy()
                del input_dict['protein_list'], input_dict['split']
                input_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == train_split).values).bool()
                input_dict['label'] = input_dict[label_fields[0]] # Currently only support annotating one label
                for k in input_dict:
                    input_dict[k] = input_dict[k].to(device)
                x_dict = XDict(input_dict)
                # print('x_dict.keys():', x_dict.keys())
                out_dict, loss, logits = self.model(x_dict, data_dict['protein_list'])
                # print('out_dict.keys():', out_dict.keys())
                # print('len(out_dict[pred]):',len(out_dict['pred']))
                # print('len(out_dict[label]):',len(out_dict['label']))
                # print('ffffffffffffiiiiiiiiiiiiiiiiiiinnnnnnnnnnnnnnnnnnnnaaaaaaaalllllllllllllllll')
                with torch.no_grad():
                    train_scores.append(
                        downstream_eval('embedding', out_dict['pred'], out_dict['label'], **self.eval_dict))  #annotation

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
                device: Union[str, torch.device] = 'cpu'
                ):
        if inference_config and 'batch_size' in inference_config:
            batch_size = inference_config['batch_size']
        else:
            batch_size = 0
        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_protein_list:
            warnings.warn('`batch_protein_list` argument is ignored in CellEmbeddingPipeline.')
        return self._inference(adata, batch_size, device, ensembl_auto_conversion)

    def _inference(self, adata: ad.AnnData,
                batch_size: int = 0,
                device: Union[str, torch.device] = 'cpu',
                ensembl_auto_conversion: bool = True):
        self.model.to(device)
        # # adata = self.common_preprocess(adata, 0, covariate_fields=None, ensembl_auto_conversion=ensembl_auto_conversion)
        # print('because we are working on proteins not transcriptmics, I commented out the line above me in cell_embedding.py')
        # print(f'After filtering, {adata.shape[1]} proteins remain.')
        dataset = TranscriptomicDataset(adata, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        # order_list = []
        if batch_size <= 0:
            batch_size = adata.shape[0]

        # with torch.no_grad():
        #     self.model.eval()
        #     pred = []
        #     for i, data_dict in enumerate(dataloader):
        #         idx = torch.arange(data_dict['x_seq'].shape[0])
        #         for j in range(0, len(idx), batch_size):
        #             if len(idx) - j < batch_size:
        #                 cur = idx[j:]
        #             else:
        #                 cur = idx[j:j + batch_size]
        #             input_dict = {}
        #             for k in data_dict:
        #                 if k == 'x_seq':
        #                     input_dict[k] = data_dict[k].index_select(0, cur).to(device)
        #                 elif k not in ['protein_list', 'split']:
        #                     input_dict[k] = data_dict[k][cur].to(device)
        #             x_dict = XDict(input_dict)
        #             print('x_dict.keys() in def _inference:', x_dict.keys())
        #             out_dict, _ = self.model(x_dict, data_dict['protein_list']) 
        #             order_list.append(input_dict['order_list'])
        #             pred.append(out_dict['pred'])#[input_dict['order_list']])
        #     order = torch.cat(order_list)
        #     order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
        #     pred = torch.cat(pred)
        #     pred = pred[order]
        #     return pred

        with torch.no_grad():
            self.model.eval()
            epoch_loss = []
            order_list = []
            pred = []
            # label = []
            for i, data_dict in enumerate(dataloader):
                # if not order_required and split and np.sum(data_dict['split'] == split) == 0:
                    # continue
    
                idx = torch.arange(data_dict['x_seq'].shape[0])
                # if split:
                    # data_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == split).values).bool()
                # else:
                data_dict['loss_mask'] = torch.ones(data_dict['x_seq'].shape[0]).bool()
                # if label_fields:
                #     data_dict['label'] = data_dict[label_fields[0]]
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
                    out_dict, loss, logits = self.model(x_dict, data_dict['protein_list'])
                    # if 'label' in input_dict:
                    #     epoch_loss.append(loss.item())
                    #     label.append(out_dict['label'])
                    # if order_required:
                    #     order_list.append(input_dict['order_list'])
                    epoch_loss.append(loss.item())
                    pred.append(out_dict['pred'])
    
            pred = torch.cat(pred)
            print('pred after cat:', pred)
            return {'pred': pred, 'logits': logits}
            # if order_required:
            #     order = torch.cat(order_list)
            #     order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            #     pred = pred[order]
    
            if len(epoch_loss) == 0:
                return {'pred': pred}

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_protein_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu'
              ):
        if evaluation_config and 'batch_size' in evaluation_config:
            batch_size = evaluation_config['batch_size']
        else:
            batch_size = 0
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for evaluation of cell embedding pipeline. Please raise an issue on Github for further support.')
        if split_field:
            warnings.warn('`split_field` argument is ignored in CellEmbeddingPipeline.')
        if target_split:
            warnings.warn('`target_split` argument is ignored in CellEmbeddingPipeline.')
        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_protein_list:
            warnings.warn('`batch_protein_list` argument is ignored in CellEmbeddingPipeline.')

        adata = adata.copy()
        pred = self._inference(adata, batch_size, device)
        adata.obsm['emb'] = pred.cpu().numpy()
        if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
            sc.pp.neighbors(adata, use_rep='emb', method='rapids')
        else:
            sc.pp.neighbors(adata, use_rep='emb')
        best_ari = -1
        best_nmi = -1
        for res in range(1, 15, 1):
            res = res / 10
            if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
                import rapids_singlecell as rsc
                rsc.tl.leiden(adata, resolution=res, key_added='leiden')
            else:
                sc.tl.leiden(adata, resolution=res, key_added='leiden')
            ari_score = adjusted_rand_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if ari_score > best_ari:
                best_ari = ari_score
            nmi_score = normalized_mutual_info_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if nmi_score > best_nmi:
                best_nmi = nmi_score
        return {'ari': best_ari, 'nmi': best_nmi}




