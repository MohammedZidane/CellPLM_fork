import os
import torch
import anndata as ad
from ..model import OmicsFormer
from abc import ABC, abstractmethod
from typing import List, Union
from .experimental import symbol_to_ensembl
import json
import warnings
import scanpy as sc

def load_pretrain(
        pretrain_prefix: str,
        overwrite_config: dict = None,
        pretrain_directory: str = './ckpt'):
    print('in the load_pretrain function')
    config_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.config.json')
    print('config_path:', config_path)
    ckpt_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.best.ckpt')
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
    config.update(overwrite_config)
    model = OmicsFormer(**config)
    pretrained_model_dict = torch.load(ckpt_path)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_model_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def load_untrained_model(
        pretrain_prefix: str,
        overwrite_config: dict = None,
        pretrain_directory: str = './ckpt'):
    print('in the load_untrained_model function')
    # Path to the configuration file.
    config_path = os.path.join(pretrain_directory, f'{pretrain_prefix}.config.json')
    print('config_path:', config_path)

    # Load the model configuration.
    with open(config_path, "r") as openfile:
        config = json.load(openfile)
    
    # If there are any configurations to overwrite, update the config dictionary.
    if overwrite_config:
        config.update(overwrite_config)

    # Initialize a new instance of the model with the loaded (or overwritten) configuration.
    model = OmicsFormer(**config)

    # The model is now initialized but not trained (weights are not loaded).

    return model

class Pipeline(ABC):
    def __init__(self,
                 pretrain_prefix: str,
                 overwrite_config: dict = None,
                 pretrain_directory: str = './ckpt',
                 pretrain = True,
                 ):
        # Load pretrain model
        if not pretrain:
            self.model = load_pretrain(pretrain_prefix, overwrite_config, pretrain_directory)
        else:
            self.model = load_untrained_model(pretrain_prefix, overwrite_config, pretrain_directory)
        self.protein_list = None
        self.fitted = False
        self.eval_dict = {}

    def common_preprocess(self, adata, hvg, covariate_fields, ensembl_auto_conversion):
        if covariate_fields:
            for i in covariate_fields:
                assert i in ['batch', 'dataset',
                             'platform'], 'Currently does not support customized covariate other than "batch", "dataset" and "platform"'
        adata = adata.copy()
        if not adata.var.index.isin(self.model.protein_set).any():
            if ensembl_auto_conversion:
                print('Automatically converting protein symbols to ensembl ids...')
                adata.var.index = symbol_to_ensembl(adata.var.index.tolist())
                if (adata.var.index == '0').all():
                    raise ValueError(
                        'None of AnnData.var.index found in pre-trained protein set.')
                adata.var_names_make_unique()
            else:
                raise ValueError(
                    'None of AnnData.var.index found in pre-trained protein set. In case the input protein names are protein symbols, please enable `ensembl_auto_conversion`, or manually convert protein symbols to ensembl ids in the input dataset.')
        if self.fitted:
            return adata[:, adata.var.index.isin(self.protein_list)]
        else:
            if hvg > 0:
                if hvg < adata.shape[1]:
                    sc.pp.highly_variable_prs(adata, n_top_genes=hvg, subset=True, flavor='seurat_v3')
                else:
                    warnings.warn('HVG number is larger than number of valid genes.')
            adata = adata[:, [x for x in adata.var.index.tolist() if x in self.model.gene_set]]
            self.gene_list = adata.var.index.tolist()
            return adata

    @abstractmethod
    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None, # A field in adata.obs for representing train-test split
            train_split: str = None, # A specific split where labels can be utilized for training
            valid_split: str = None, # A specific split where labels can be utilized for validation
            covariate_fields: List[str] = None, # A list of fields in adata.obs that contain cellular covariates
            label_fields: List[str] = None, # A list of fields in adata.obs that contain cell labels
            batch_protein_list: dict = None,  # A dictionary that contains batch and protein list pairs
            ensembl_auto_conversion: bool = True, # A bool value indicating whether the function automativally convert symbols to ensembl id
            device: Union[str, torch.device] = 'cpu'
            ):
        # Fine-tune the model on an anndata object
        pass

    @abstractmethod
    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_protein_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu'
                ):
        # Inference on an anndata object
        pass

    @abstractmethod
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
        # Inference on an anndata object and automatically evaluate
        pass