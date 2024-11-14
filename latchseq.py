from hflayers.auxiliary.data import LatchSequenceSet
from torch.utils.data import DataLoader

latch_sequence_set = LatchSequenceSet(num_samples=2, num_instances=3, num_characters= 4)

data_loader = DataLoader(dataset=latch_sequence_set, batch_size=1) 

for sample_data in data_loader:
    print(sample_data[r'data'], sample_data[r'target'])
