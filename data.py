import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from digitize import Digitize
# from sklearn.model_selection import train_test_split

class SentenceDataset(Dataset):
    def __init__(self, file_path: str, padding: int = 512):
        self.file_path = file_path
        self.padding = padding
        self.sentences = None
        self.dataset = None
        self._load_data()
        self._prepare_dataset()

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        self.sentences = df["ENGLISH"]

    def _prepare_dataset(self):
        self.dataset = [
            {
                'data': torch.tensor(Digitize(sentence, padding=self.padding).encode(), dtype=torch.long),
                'target': torch.tensor(Digitize(sentence, padding=self.padding).encode(), dtype=torch.long)
            }
            for sentence in self.sentences
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

if __name__ == "__main__":
    dataset = SentenceDataset(file_path="1000sents.csv", padding=512)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    data_loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data_loader_eval = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    for batch in data_loader_train:
        print("Train batch data:", batch['data'])
        print("Train batch target:", batch['target'])
        break

    for batch in data_loader_eval:
        print("Eval batch data:", batch['data'])
        print("Eval batch target:", batch['target'])
        break

