from tqdm import tqdm
import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
from networks import MultiLayerPerceptron, ConvNet, get_nn_params, flatten_params, recon_params
from data import gen_infimnist, MyDataset
import torch.nn.functional as F
from torch import nn, optim, hub
from comm import *
from dis_dist import add_gauss, add_binom, add_gauss_slow
from autodp import rdp_bank, rdp_acct, dp_acct, privacy_calibrator
import pickle as pkl

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

RESULT_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f.pkl'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--datasize', type=int, default=10000000)
    parser.add_argument('--nworker', type=int, default=100000)
    parser.add_argument('--perround', type=int, default=100)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=10)
    # 'homo' or 'hetero'
    parser.add_argument('--dist', default='homo')
    parser.add_argument('--checkpoint', type=int, default=1)
    # L2 Norm bound for clipping gradient
    parser.add_argument('--clipbound', type=float, default=0.25)
    # The number of levels for quantization and the L_inf bound for quantization
    # FIXME: np.sqrt(d)
    parser.add_argument('--quanlevel', type=int, default=2*100+1)
    # The size of the additive group used in secure aggregation
    # parser.add_argument('--cylicbound', type=float, default=3.)
    # The variance of the discrete Gaussian noise
    # non-private, dis-gauss, and binom
    parser.add_argument('--dp', default='dis-gauss')
    parser.add_argument('--deltatot', type=float, default=1e-5)
    parser.add_argument('--sigma2', type=float, default=0.25)
    # parser.add_argument('--eps', type=float, default=1.)
    parser.add_argument('--delta', type=float, default=1e-6)
    parser.add_argument('--momentum')
    parser.add_argument('--weightdecay')
    parser.add_argument('--debug', default='n')
    args = parser.parse_args()

    DEVICE = "cuda:" + args.device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    DATASET = args.dataset
    SIZE = args.datasize
    NWORKER = args.nworker
    PERROUND = args.perround
    SUBSAMPLING_RATE = float(PERROUND) / NWORKER
    LOCALITER = args.localiter
    EPOCH = args.epoch
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batchsize
    params = {'batch_size': BATCH_SIZE, 'shuffle': True}
    CHECK_POINT = args.checkpoint
    CLIP_BOUND = args.clipbound
    QUANTIZE_LEVEL = args.quanlevel
    QUANTIZE_BOUND = CLIP_BOUND
    INTERVAL = 2 * QUANTIZE_BOUND / (QUANTIZE_LEVEL-1)
    Q = INTERVAL / 2
    S = 1
    DP = args.dp
    SIGMA2 = args.sigma2
    SENSITIVITY  = 4 * CLIP_BOUND
    # M = max(23*np.log(10*/args.delta))# 8 * np.log(2./args.delta) / args.eps / args.eps
    P = 0.5
    # EPS = np.log(1+SUBSAMPLING_RATE * (np.exp(args.eps)-1))
    DELTA = 3 * SUBSAMPLING_RATE * args.delta
    DELTA_TOT = args.deltatot
    DEBUG = args.debug
    S = 1

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        # read in the dataset with numpy array split them and then use data loader to wrap them
        train_set = MyDataset(FEATURE_TEMPLATE%('train',0,SIZE), TARGET_TEMPLATE%('train',0,SIZE), transform=transform)
        test_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%('test',0,10000), TARGET_TEMPLATE%('test',0,10000), transform=transform), batch_size=BATCH_SIZE)


        network = MultiLayerPerceptron().to(device)

    elif DATASET == 'CIFAR10':

        transform = torchvision.transforms.Compose([
                                         torchvision.transforms.CenterCrop(24), 
                                         torchvision.transforms.ToTensor(), 
                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_loader = DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform))

        network = ConvNet().to(device)

    # generate random rotation matrix
    plain_size, param_size = get_nn_params(network)
    DIAG = random_diag(param_size)
    SENSINF = QUANTIZE_LEVEL + 1
    SENS1 = np.sqrt(plain_size) * CLIP_BOUND / Q + np.sqrt(2 * np.sqrt(plain_size) * CLIP_BOUND * np.log(2 / args.delta) / Q) + 4 * np.log(2 / args.delta) / 3
    SENS2 = CLIP_BOUND / Q + np.sqrt(SENS1 + np.sqrt(2 * np.sqrt(plain_size) * CLIP_BOUND * np.log(2 / args.delta) / Q))
    M = int(1 / P / (1-P) * max(23*np.log(10*plain_size/args.delta), 2*SENSINF / INTERVAL))
    EPS_ = SENS2 * np.sqrt(2 * np.log(1.25/args.delta)) / S / np.sqrt(M*P*(1-P)) +(SENS2 * 5 * np.sqrt(np.log(10/args.delta)) / 2 + SENS1 / 3) / S / M / P / (1-P) / (1-args.delta/10)  + (2 * SENSINF * np.log(1.25/args.delta) / 3 + 2 * SENSINF * np.log(20*plain_size/args.delta) * np.log(10/args.delta) / 3) / S / M / P / (1-P)
    EPS = np.log(1+SUBSAMPLING_RATE * (np.exp(EPS_)-1))
    NBIT = np.ceil(np.log2(M + QUANTIZE_LEVEL))
    CYLIC_BOUND = 2**NBIT
    CYLIC_LEVEL = int(CYLIC_BOUND / INTERVAL + 1)

    # Split into multiple training set
    TRAIN_SIZE = len(train_set) // NWORKER
    assert TRAIN_SIZE > 0, "Each worker should have at least one data point!"
    # print(len(train_set), NWORKER, TRAIN_SIZE, BATCH_SIZE)
    train_loaders = []
    if args.dist == 'homo':
        sizes = []
        sum = 0
        for i in range(0, NWORKER):
            sizes.append(TRAIN_SIZE)
            sum = sum + TRAIN_SIZE
        sizes[0] = sizes[0] + len(train_set)  - sum
        train_sets = random_split(train_set, sizes)
        for trainset in train_sets:
            train_loaders.append(DataLoader(trainset, **params))

    elif args.dist == 'hetero':
        # FIXME: add CIFAR10 case
        idx = train_set.target==0
        feature = train_set.feature[idx]
        target = train_set.target[idx]
        for i in range(1, 10):
            idx = train_set.target==i
            feature = np.concatenate((feature, train_set.feature[idx]), 0)
            target = np.concatenate((target, train_set.target[idx]), 0)
        feature = torch.Tensor(feature)
        target = torch.Tensor(target)
        train_set = TensorDataset(feature, target)
        for i in range(0, NWORKER):
            train_loaders.append(DataLoader(Subset(train_set, range(i*TRAIN_SIZE, (i+1)*TRAIN_SIZE)), **params))

    # define training loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    # prepare data structures to store local gradients
    local_grads = {}
    for i in range(NWORKER):
        local_grads[i] = np.zeros(param_size)

    # define performance metrics
    ups = 0

    # analytical moments accountant
    if DP == 'dis-gauss':
        acct = rdp_acct.anaRDPacct()
        func_gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': np.sqrt(SIGMA2) / SENSITIVITY}, x)

    results = {'privacy':[], 'accuracy':[]}

    for epoch in range(EPOCH):
        # select workers per subset 
        print("Epoch: ", epoch)
        choices = np.random.choice(NWORKER, PERROUND)
        # print(choices)
        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
           params_copy.append(p.clone())
        params_flat_copy = flatten_params(list(network.parameters()), param_size)
        for c in tqdm(choices):
            # print(c)
            for iepoch in range(0, LOCALITER):
                for idx, (feature, target) in enumerate(train_loaders[c], 0):
                    feature = feature.view(-1, 28*28).to(device)
                    target = target.type(torch.long).to(device)
                    optimizer.zero_grad()
                    output = network(feature)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # compute the difference
            local_grads[c] = params_flat_copy - flatten_params(network.parameters(), param_size)
            local_grads[c] = clip_gradient(local_grads[c], CLIP_BOUND)
            if DEBUG == 'n':
                local_grads[c] = rotate(local_grads[c], DIAG)
                local_grads[c] = quantize(local_grads[c], QUANTIZE_LEVEL, QUANTIZE_BOUND)
            if DP == 'dis-gauss':
                if DEBUG == 'n':
                    local_grads[c] = add_gauss(local_grads[c], SIGMA2 / PERROUND, INTERVAL)
                else:
                    local_grads[c] += np.random.normal(0., np.sqrt(SIGMA2 / PERROUND), size=len(local_grads[c]))
                local_grads[c] = cylicRound(local_grads[c], CYLIC_LEVEL, CYLIC_BOUND)
            elif DP == 'binom':
                local_grads[c] = add_binom(local_grads[c], M / PERROUND, P, INTERVAL)
            
            # manually restore the parameters of the global network
            with torch.no_grad():
                for idx, p in enumerate(list(network.parameters())):
                    p.copy_(params_copy[idx])

        # aggregation
        average_grad = np.zeros(param_size)
        for c in choices:
            average_grad = average_grad + local_grads[c] / PERROUND

        params = list(network.parameters())
        if DEBUG == 'n':
            average_grad = rotate(average_grad, DIAG, reverse=True)
        average_grad = recon_params(average_grad, network)
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)
        if DP == 'dis-gauss':
            acct.compose_subsampled_mechanism(func_gaussian, SUBSAMPLING_RATE)

        if (epoch+1) % CHECK_POINT == 0:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.cross_entropy(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            if DP == 'dis-gauss':
                eps = acct.get_eps(DELTA_TOT)
            elif DP == 'binom':
                eps = np.sqrt(2*(epoch+1)*np.log(1./(DELTA_TOT-epoch*DELTA)))*EPS + (epoch+1)*EPS*(np.exp(EPS)-1)
            if DP != 'non-private':
                results['privacy'].append(eps)
            results['accuracy'].append((100. * correct / len(test_loader.dataset)).data)
            if DP != 'non-private':
                print('\nEpoch: {}, Epsilon: {:.4f}\n'.format(epoch, eps))
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    if DP == 'dis-gauss':
        output = open(RESULT_TEMPLATE%(DATASET, DP, args.dist, QUANTIZE_LEVEL, NBIT, SIGMA2), 'wb')
    elif DP == 'binom':
        output = open(RESULT_TEMPLATE%(DATASET, DP, args.dist, QUANTIZE_LEVEL, NBIT, 0), 'wb')
    else:
        output = open(RESULT_TEMPLATE%(DATASET, DP, args.dist, 0, 0, 0), 'wb')
    pkl.dump(results, output)
    output.close()
