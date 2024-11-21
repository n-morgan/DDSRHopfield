from torch.utils.data import DataLoader

from dataloader import MnistBags

data_loader_train = DataLoader(MnistBags(
    target_number=9,
    mean_bag_length=10,
    var_bag_length=2,
    num_bag=200,
    train=True
), batch_size=1, shuffle=True)

# Create data loader of validation set.
data_loader_eval = DataLoader(MnistBags(
    target_number=9,
    mean_bag_length=10,
    var_bag_length=2,
    num_bag=50,
    train=False
), batch_size=1, shuffle=True)



for sample_data in data_loader_train:
    print(sample_data[r'data'], sample_data[r'target'])
