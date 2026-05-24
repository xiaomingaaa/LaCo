import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset
import pandas as pd

class MyInMemoryDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 Sdrugs_data,
                 Scelllines_data,
                 dgi_data=None,              
                 transform=None,
                 pre_transform=None,
                 args = None,
                 max_node_num=155):

        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)


        if args.celldataset == 1:
            self.name = osp.basename(data_items).split('items')[0]+'18498g'
        elif args.celldataset == 2:
            self.name = osp.basename(data_items).split('items')[0]+'4079g'
        elif args.celldataset == 3:
            self.name = osp.basename(data_items).split('items')[0]+'963g'

        self.name = self.name+'_TransDrug_norm'
        
        if args.mode == 'infer':
            self.name = osp.basename(data_items).split('items')[0]
        

        self.args = args
        self.data_items = pd.read_csv(data_items) 
        self.celllines = np.load(celllines_data, allow_pickle=True)
        self.scelllines = np.load(Scelllines_data, allow_pickle=True).item()
        self.drugs = np.load(drugs_data, allow_pickle=True)
        self.sdrugs = np.load(Sdrugs_data, allow_pickle=True).item()
        if dgi_data: 
            self.dgi = np.load(dgi_data, allow_pickle=True).item()
        else:
            self.dgi = {}
        self.max_node_num = max_node_num

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir) 

    def process(self):  
        data_list = []
        data_len = len(self.data_items)

        for i in tqdm(range(data_len)):
            
            drugA = self.data_items.iloc[i]['drug_a']
            drugB = self.data_items.iloc[i]['drug_b']
            cell = self.data_items.iloc[i]['cell']
            label = self.data_items.iloc[i]['synergy'] 
            drug_map = pd.read_csv('MyDataset/LLM_data/drug_map.csv')
            cell_map = pd.read_csv('MyDataset/LLM_data/cell_map.csv')
            sdrug_map = pd.read_csv('MyDataset/drug_map_with_smiles.csv')
            scell_map = pd.read_csv('MyDataset/cell.csv')
            
            drugA_index = drug_map.loc[drug_map['name'] == drugA, 'index'].values[0]
            drugB_index = drug_map.loc[drug_map['name'] == drugB, 'index'].values[0]
            cell_index = cell_map.loc[cell_map['name'] == cell, 'index'].values[0]
            
            sdrugA_index = sdrug_map.loc[sdrug_map['name'] == drugA, 'processed_smiles'].values[0]
            sdrugB_index = sdrug_map.loc[sdrug_map['name'] == drugB, 'processed_smiles'].values[0]
            scell_index = scell_map.loc[scell_map['name'] == cell, 'depmap_id'].values[0]

            cell_features = self.celllines[cell_index]
            scell_features = self.scelllines[scell_index]
            dgiA = self.dgi.get(drugA, np.ones(cell_features.shape[0]))
            dgiB = self.dgi.get(drugB, np.ones(cell_features.shape[0]))
            drugA_features = self.drugs[drugA_index]
            drugB_features = self.drugs[drugB_index]
            sdrugA_features = self.sdrugs[sdrugA_index]
            sdrugB_features = self.sdrugs[sdrugB_index]
            cell_drug_data = Data()
            cell_drug_data.drugA = torch.Tensor(np.array([drugA_features])).to(dtype=torch.float32)
            cell_drug_data.drugB = torch.Tensor(np.array([drugB_features])).to(dtype=torch.float32)
            cell_drug_data.x_cell = torch.Tensor(np.array([cell_features])).to(dtype=torch.float32)
            # cell_drug_data.x_cell = torch.as_tensor(cell_features).to(dtype=torch.float32)
            cell_drug_data.sdrugA = torch.Tensor(np.array([sdrugA_features])).to(dtype=torch.float32)
            cell_drug_data.sdrugB = torch.Tensor(np.array([sdrugB_features])).to(dtype=torch.float32)
            cell_drug_data.sx_cell = torch.as_tensor(scell_features).to(dtype=torch.float32)
            
            
            cell_drug_data.y = torch.Tensor([float(label)]).to(dtype=torch.float32)

            cell_drug_data.dgiA = torch.Tensor(dgiA).to(dtype=torch.float32)
            cell_drug_data.dgiB = torch.Tensor(dgiB).to(dtype=torch.float32)
            data_list.append(cell_drug_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list) 
        torch.save((data, slices), self.processed_paths[0])

        print('Dataset construction done.')
