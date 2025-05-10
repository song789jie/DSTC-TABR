import torch
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
import json
import numpy as np


class Haptic_Dataset(Dataset):
    def __init__(
            self,
            root_path='../dataset',
            data_path='norm_DFT321_data_len64',
            json_name='data_path_len64.json',
    ):
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        self.json_name = json_name
        self.root_path_folder = Path(root_path)
        self.data_path_folder = Path(root_path, data_path)
        self.json_file = Path(root_path, json_name)
        assert self.root_path_folder.exists(), 'root folder does not exist'
        assert self.data_path_folder.exists(), 'data folder does not exist'

        with open(self.json_file, 'r') as json_file:
            self.list_data_path = json.load(json_file)
            # print(self.list_data_path)

    def __len__(self):
        return len(self.list_data_path)

    def __getitem__(self, idx):
        file_path = self.list_data_path[idx]
        loaded_list = np.loadtxt(Path(self.root_path, self.data_path, file_path))
        loaded_tensor = torch.from_numpy(loaded_list)
        loaded_tensor = loaded_tensor.unsqueeze(0)
        return loaded_tensor


class Haptic_DataLoader:
    def __init__(
            self,
            dataset: Haptic_Dataset,
            batch_size,
            num_workers,
            vali_dataset=None,
            shuffle=True,
            valid_frac=0.05,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vali_dataset = vali_dataset
        self.shuffle = shuffle
        self.valid_frac = valid_frac
        self.dataset = dataset
        self.random_split_seed = 42

        if self.vali_dataset is not None:
            self.train_dataset = self.dataset
            self.valid_dataset = self.vali_dataset
            print(f' 5_fold_crossvalidation training with dataset of {len(self.train_dataset)} samples'
                  f' and validating with randomly splitted {len(self.valid_dataset)} samples')

        elif valid_frac > 0:
            train_size = int((1 - valid_frac) * dataset.__len__())
            valid_size = dataset.__len__() - train_size
            self.train_dataset, self.valid_dataset = random_split(self.dataset,
                                                                  [train_size, valid_size],
                                                                  generator=torch.Generator()
                                                                  .manual_seed(self.random_split_seed))
            print(f'training with dataset of {len(self.train_dataset)} samples'
                  f' and validating with randomly splitted {len(self.valid_dataset)} samples')

        else:
            self.train_dataset = self.dataset
            self.valid_dataset = self.dataset
            print(f'training with shared training and valid dataset of {dataset.__len__()} samples')

    def get_dataloader(self):
        train_dataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      shuffle=True, drop_last=True)
        valid_dataLoader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      shuffle=True, drop_last=True)
        return train_dataLoader, valid_dataLoader


if __name__ == '__main__':
    path = '../dataset'
    folder = 'DFT321_data_len64'
    name = 'data_path_len64.json'
    HD = Haptic_Dataset(root_path=path, data_path=folder, json_name=name)
    print(HD.__len__())
    print(HD.__getitem__(1).shape)
