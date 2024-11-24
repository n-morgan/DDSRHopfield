from digitize import Digitize
import pandas as pd
import torch
from torch import nn
import numpy as np

df = pd.read_csv("1000sents.csv")
sentences = df["ENGLISH"]

sentence = next(iter(sentences))

tensor =  torch.tensor(np.array(Digitize(sentence, 15).encode())[np.newaxis, : ] , dtype=torch.long)

print(f"Target: {tensor[0]}, Target dtype: {tensor[0].dtype}")


# tensor =  torch.tensor(np.array([1,2,3])[:, np.newaxis], dtype=torch.long)








print(tensor)
