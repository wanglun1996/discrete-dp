import argparse
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from networks import MultiLayerPerceptron
from data import gen_infimnist, MyDataset

FEATURE_TEMPLATE = '../data/infimnist_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_target_%d_%d.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--nworker', type=int)
    parser.add_argument('--perround', type=int)
    parser.add_argument('--localiter', type=int)
    parser.add_argument('--up_bit')
    parser.add_argument('--down_bit')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum')
    parser.add_argument('--weightdecay')
    parser.add_argument('--network')
    args = parser.parse_args()

    DEVICE = "cuda:" + args.device
    DATASET = args.dataset
    NWORKER = args.nworker
    PERROUND = args.perround
    LOCALITER = args.localiter
    LEARNING_RATE = args.lr

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])
        batch_size = 5
        dataset_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%(0,100), TARGET_TEMPLATE%(0,100), transform=transform), batch_size=batch_size, shuffle=True)

        network = MultiLayerPerceptron()
        #TODO: do we need momentum?
        optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)
        
        examples = enumerate(dataset_loader)
        batch_idx, (feature, target) = next(examples)
        print(feature)

        
