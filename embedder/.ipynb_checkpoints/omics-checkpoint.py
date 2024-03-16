import torch
from torch import nn
import torch.nn.functional as F
from ..utils.pe import select_pe_encoder
from ..utils import create_norm, create_activation
import numpy as np
from ..utils.sparse import sparse_normalize, sparse_tpm

class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_protein_list, num_hid, protein_emb=None, fix_embedding=False):
        super().__init__()
        print('pretrained_protein_list0000000000:', pretrained_protein_list)
        self.pretrained_protein_list = pretrained_protein_list
        self.protein_index = dict(zip(pretrained_protein_list, list(range(len(pretrained_protein_list)))))
        self.num_hid = num_hid

        if protein_emb is not None:
            print('pretrained_protein_list111111:', pretrained_protein_list)
            self.emb = nn.Parameter(protein_emb, requires_grad=not fix_embedding)
        else:
            self.emb = nn.Parameter(torch.randn([len(pretrained_protein_list), num_hid], dtype=torch.float32)*0.005)
            print('pretrained_protein_list222222222:', pretrained_protein_list)
            if fix_embedding:
                self.emb.requires_grad = False

    def forward(self, x_dict, input_protein_list=None):
        if 'masked_x_seq' in x_dict:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']

        if 'dropout' in x_dict:
            indices = x._indices().t()
            values = x._values()
            temp = values.sum()
            values = values.float()
            values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
            # x = torch.sparse.FloatTensor(indices.t(), values, x.shape) #NO NEED FOR SPARSE REPRESENTATOIN

        x = torch.log1p(x)
        # x = sparse_tpm(x)
        if input_protein_list is not None:
            input_protein_list = ['CD44 - stroma:Cyc_2_ch_2', 'FOXP3 - regulatory T cells:Cyc_2_ch_3', 'CD8 - cytotoxic T cells:Cyc_3_ch_2', 'p53 - tumor suppressor:Cyc_3_ch_3', 'GATA3 - Th2 helper T cells:Cyc_3_ch_4', 'CD45 - hematopoietic cells:Cyc_4_ch_2', 'T-bet - Th1 cells:Cyc_4_ch_3', 'beta-catenin - Wnt signaling:Cyc_4_ch_4', 'HLA-DR - MHC-II:Cyc_5_ch_2', 'PD-L1 - checkpoint:Cyc_5_ch_3', 'Ki67 - proliferation:Cyc_5_ch_4', 'CD45RA - naive T cells:Cyc_6_ch_2', 'CD4 - T helper cells:Cyc_6_ch_3', 'CD21 - DCs:Cyc_6_ch_4', 'MUC-1 - epithelia:Cyc_7_ch_2', 'CD30 - costimulator:Cyc_7_ch_3', 'CD2 - T cells:Cyc_7_ch_4', 'Vimentin - cytoplasm:Cyc_8_ch_2', 'CD20 - B cells:Cyc_8_ch_3', 'LAG-3 - checkpoint:Cyc_8_ch_4', 'Na-K-ATPase - membranes:Cyc_9_ch_2', 'CD5 - T cells:Cyc_9_ch_3', 'IDO-1 - metabolism:Cyc_9_ch_4', 'Cytokeratin - epithelia:Cyc_10_ch_2', 'CD11b - macrophages:Cyc_10_ch_3', 'CD56 - NK cells:Cyc_10_ch_4', 'aSMA - smooth muscle:Cyc_11_ch_2', 'BCL-2 - apoptosis:Cyc_11_ch_3', 'CD25 - IL-2 Ra:Cyc_11_ch_4', 'CD11c - DCs:Cyc_12_ch_3', 'PD-1 - checkpoint:Cyc_12_ch_4', 'Granzyme B - cytotoxicity:Cyc_13_ch_2', 'EGFR - signaling:Cyc_13_ch_3', 'VISTA - costimulator:Cyc_13_ch_4', 'CD15 - granulocytes:Cyc_14_ch_2', 'ICOS - costimulator:Cyc_14_ch_4', 'Synaptophysin - neuroendocrine:Cyc_15_ch_3', 'GFAP - nerves:Cyc_16_ch_2', 'CD7 - T cells:Cyc_16_ch_3', 'CD3 - T cells:Cyc_16_ch_4', 'Chromogranin A - neuroendocrine:Cyc_17_ch_2', 'CD163 - macrophages:Cyc_17_ch_3', 'CD45RO - memory cells:Cyc_18_ch_3', 'CD68 - macrophages:Cyc_18_ch_4', 'CD31 - vasculature:Cyc_19_ch_3', 'Podoplanin - lymphatics:Cyc_19_ch_4', 'CD34 - vasculature:Cyc_20_ch_3', 'CD38 - multifunctional:Cyc_20_ch_4', 'CD138 - plasma cells:Cyc_21_ch_3', 'HOECHST1:Cyc_1_ch_1', 'CDX2 - intestinal epithelia:Cyc_2_ch_4', 'Collagen IV - bas. memb.:Cyc_12_ch_2', 'CD194 - CCR4 chemokine R:Cyc_14_ch_3', 'MMP9 - matrix metalloproteinase:Cyc_15_ch_2', 'CD71 - transferrin R:Cyc_15_ch_4', 'CD57 - NK cells:Cyc_17_ch_4', 'MMP12 - matrix metalloproteinase:Cyc_21_ch_4', 'DRAQ5:Cyc_23_ch_4']

            self.protein_index = {'CD44 - stroma:Cyc_2_ch_2': 0,
         'FOXP3 - regulatory T cells:Cyc_2_ch_3': 1,
         'CD8 - cytotoxic T cells:Cyc_3_ch_2': 2,
         'p53 - tumor suppressor:Cyc_3_ch_3': 3,
         'GATA3 - Th2 helper T cells:Cyc_3_ch_4': 4,
         'CD45 - hematopoietic cells:Cyc_4_ch_2': 5,
         'T-bet - Th1 cells:Cyc_4_ch_3': 6,
         'beta-catenin - Wnt signaling:Cyc_4_ch_4': 7,
         'HLA-DR - MHC-II:Cyc_5_ch_2': 8,
         'PD-L1 - checkpoint:Cyc_5_ch_3': 9,
         'Ki67 - proliferation:Cyc_5_ch_4': 10,
         'CD45RA - naive T cells:Cyc_6_ch_2': 11,
         'CD4 - T helper cells:Cyc_6_ch_3': 12,
         'CD21 - DCs:Cyc_6_ch_4': 13,
         'MUC-1 - epithelia:Cyc_7_ch_2': 14,
         'CD30 - costimulator:Cyc_7_ch_3': 15,
         'CD2 - T cells:Cyc_7_ch_4': 16,
         'Vimentin - cytoplasm:Cyc_8_ch_2': 17,
         'CD20 - B cells:Cyc_8_ch_3': 18,
         'LAG-3 - checkpoint:Cyc_8_ch_4': 19,
         'Na-K-ATPase - membranes:Cyc_9_ch_2': 20,
         'CD5 - T cells:Cyc_9_ch_3': 21,
         'IDO-1 - metabolism:Cyc_9_ch_4': 22,
         'Cytokeratin - epithelia:Cyc_10_ch_2': 23,
         'CD11b - macrophages:Cyc_10_ch_3': 24,
         'CD56 - NK cells:Cyc_10_ch_4': 25,
         'aSMA - smooth muscle:Cyc_11_ch_2': 26,
         'BCL-2 - apoptosis:Cyc_11_ch_3': 27,
         'CD25 - IL-2 Ra:Cyc_11_ch_4': 28,
         'CD11c - DCs:Cyc_12_ch_3': 29,
         'PD-1 - checkpoint:Cyc_12_ch_4': 30,
         'Granzyme B - cytotoxicity:Cyc_13_ch_2': 31,
         'EGFR - signaling:Cyc_13_ch_3': 32,
         'VISTA - costimulator:Cyc_13_ch_4': 33,
         'CD15 - granulocytes:Cyc_14_ch_2': 34,
         'ICOS - costimulator:Cyc_14_ch_4': 35,
         'Synaptophysin - neuroendocrine:Cyc_15_ch_3': 36,
         'GFAP - nerves:Cyc_16_ch_2': 37,
         'CD7 - T cells:Cyc_16_ch_3': 38,
         'CD3 - T cells:Cyc_16_ch_4': 39,
         'Chromogranin A - neuroendocrine:Cyc_17_ch_2': 40,
         'CD163 - macrophages:Cyc_17_ch_3': 41,
         'CD45RO - memory cells:Cyc_18_ch_3': 42,
         'CD68 - macrophages:Cyc_18_ch_4': 43,
         'CD31 - vasculature:Cyc_19_ch_3': 44,
         'Podoplanin - lymphatics:Cyc_19_ch_4': 45,
         'CD34 - vasculature:Cyc_20_ch_3': 46,
         'CD38 - multifunctional:Cyc_20_ch_4': 47,
         'CD138 - plasma cells:Cyc_21_ch_3': 48,
         'HOECHST1:Cyc_1_ch_1': 49,
         'CDX2 - intestinal epithelia:Cyc_2_ch_4': 50,
         'Collagen IV - bas. memb.:Cyc_12_ch_2': 51,
         'CD194 - CCR4 chemokine R:Cyc_14_ch_3': 52,
         'MMP9 - matrix metalloproteinase:Cyc_15_ch_2': 53,
         'CD71 - transferrin R:Cyc_15_ch_4': 54,
         'CD57 - NK cells:Cyc_17_ch_4': 55,
         'MMP12 - matrix metalloproteinase:Cyc_21_ch_4': 56,
         'DRAQ5:Cyc_23_ch_4': 57}
            print('I manually added this input_protein_list of length:', len(input_protein_list))
            # print('self.protein_index:', self.protein_index)
            protein_idx = torch.tensor([self.protein_index[o] for o in input_protein_list if o in self.protein_index]).long()
            x_dict['input_protein_mask'] = protein_idx
        else:
            if x.shape[1] != len(self.pretrained_protein_list):
                raise ValueError('The input protein size is not the same as the pretrained protein list. Please provide the input protein list.')
            protein_idx = torch.arange(x.shape[1]).long()
        protein_idx = protein_idx.to(x.device)
        # print('protein_idx:', protein_idx)
        # print('self.emb:', self.emb)
        feat = F.embedding(protein_idx, self.emb)
        feat = torch.matmul(x, feat)  # torch.sparse.mm, removed because we are NOT working with sparse (transcriptomics) data but proteomics
        return feat

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, protein_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, protein_emb=None,
                 inject_covariate=False, batch_num=None):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()#create_activation(activation)
        self.norm0 = create_norm(norm, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden),
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        if protein_emb is None:
            self.feat_enc = OmicsEmbedder(protein_list, num_emb)
        else:
            self.feat_enc = OmicsEmbedder(protein_list, num_emb, protein_emb)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False

    def forward(self, x_dict, input_protein_list=None):
        x = self.feat_enc(x_dict, input_protein_list)#self.act(self.feat_enc(x_dict, input_protein_list))
        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = 0.#self.pe_enc(pe_input)
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)
        # x = self.norm0(self.dropout(x))
        return x
