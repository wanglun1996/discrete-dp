import sys
sys.path.append('../infimnist_py')
import _infimnist as infimnist
import numpy as np
from torch.utils.data import Dataset, DataLoader

FEATURE_TEMPLATE = '../data/infimnist_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_target_%d_%d.npy'

# should I include the target in the sample?
class MyDataset(Dataset):
    def __init__(self, feature_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.target = np.load(target_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.feature[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.target[idx]
    
    def __len__(self):
        return self.target.shape[0]

def gen_infimnist(start=0, end=10000):
    mnist = infimnist.InfimnistGenerator()
    indexes = np.array(np.arange(start, end), dtype=np.int64)
    digits, labels = mnist.gen(indexes)
    digits = digits.astype(np.float32).reshape(-1, 28, 28)
    # print(digits.shape)
    np.save(FEATURE_TEMPLATE%(start, end), digits)
    np.save(TARGET_TEMPLATE%(start, end), labels)

if __name__ == '__main__':
    gen_infimnist(0, 100)
    dataset_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%(0,100), TARGET_TEMPLATE%(0,100)))
    examples = enumerate(dataset_loader)
    batch_idx, (feature, target) = next(examples)
    print(batch_idx, feature, target)
