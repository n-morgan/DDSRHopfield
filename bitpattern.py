from hflayers.auxiliary.data import BitPatternSet
from torch.utils.data import DataLoader

bit_pattern_set = BitPatternSet(
    num_bags=1,
    num_instances=16,
    num_signals=8,
    num_signals_per_bag=1,
    num_bits=8)

data_loader = DataLoader(dataset=bit_pattern_set, batch_size=1) 

for sample_data in data_loader:
    print(sample_data[r'data'], sample_data[r'target'])
