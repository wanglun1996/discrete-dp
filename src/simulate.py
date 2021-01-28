from tqdm import tqdm
import argparse
import numpy as np
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
from networks import MultiLayerPerceptron, SvhnModel, ConvNet, VGG16, get_nn_params, flatten_params, recon_params
from data import gen_infimnist, MyDataset
import torch.nn.functional as F
from torch import nn, optim, hub
from comm import *
from dis_dist import add_gauss, add_binom, add_gauss_slow, dis_gauss
from autodp import rdp_bank, rdp_acct, dp_acct, privacy_calibrator
import pickle as pkl
import copy
from torch.multiprocessing import Pool, Process, set_start_method
try:
         set_start_method('spawn')
except RuntimeError:
        pass
import functools
import gc

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

RESULT_DIS_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f-%f-%d-%d.pkl'
RESULT_BINOM_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f-%d-%d.pkl'
RESULT_BASELINE_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d.pkl'

def train(device, train_loader, network, optimizer, criterion, dp, localiter, params_flat_copy, param_size, clip_bound, sigma2, perround, cylic_level, cylic_bound, m, p, interval, debug, diag, quantize_level, quantize_bound, dataset='SVHN'):
    for iepoch in range(0, localiter):
        for idx, (feature, target) in enumerate(train_loader, 0):
            if dataset == 'INFIMNIST':
                feature = feature.view(-1, 784).to(device)
            else:
                feature = feature.to(device)
            if dataset == 'celebA':
                target = target[:, 20].type(torch.long).to(device)
            else:
                target = target.type(torch.long).to(device)

            optimizer.zero_grad()
            output = network(feature)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    local_grad = params_flat_copy - flatten_params(network.parameters(), param_size)
    local_grad = clip_gradient(local_grad, clip_bound)
    if debug == 'n' and dp != 'non-private':
        # print('here')
        local_grad = rotate(local_grad, diag)
        local_grad = quantize(local_grad, quantize_level, quantize_bound)
    if dp == 'dis-gauss':
        if debug == 'n':
            noise = dis_gauss(sigma2, interval, param_size) / perround
            local_grad += noise
        else:
            local_grad += np.random.normal(0., np.sqrt(sigma2 / perround), size=len(local_grad))
        local_grad = cylicRound(local_grad, cylic_level, cylic_bound)
    elif dp == 'binom':
        local_grad = add_binom(local_grad, m / perround, p, interval)

    del network
    del optimizer
    del criterion

    return local_grad

def smap(f):
    return f()

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
    parser.add_argument('--nbit', type=int, default=16)
    # 'homo' or 'hetero'
    parser.add_argument('--dist', default='homo')
    parser.add_argument('--checkpoint', type=int, default=1)
    # L2 Norm bound for clipping gradient
    parser.add_argument('--clipbound', type=float, default=0.25)
    # The number of levels for quantization and the L_inf bound for quantization
    # FIXME: np.sqrt(d)
    parser.add_argument('--quanlevel', type=int, default=2*10+1)
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
    QUANTIZE_BOUND = CLIP_BOUND
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
    NBIT = args.nbit
    # the following parameters might change under binomial mechanism
    QUANTIZE_LEVEL = args.quanlevel
    INTERVAL = 2 * QUANTIZE_BOUND / (QUANTIZE_LEVEL-1)
    Q = INTERVAL / 2
    CYLIC_BOUND = 2**NBIT
    CYLIC_LEVEL = int(CYLIC_BOUND / INTERVAL + 1)

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

    elif DATASET == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='../data', download=True, transform=torchvision.transforms.ToTensor())
        test_size = 12000
        train_size = len(dataset) - test_size
        torch.manual_seed(0)
        train_set, test_set = random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(test_set)

        # vgg16 = models.vgg16(pretrained=True)
        network = SvhnModel(3072, out_size=10).to(device)
        # network = torch.nn.DataParallel(SvhnModel(3072, out_size=10), device_ids=[0, 1]).to(device)
    elif DATASET == 'celebA':

        transform = torchvision.transforms.Compose([
                               torchvision.transforms.CenterCrop((178, 178)),
                               torchvision.transforms.Resize((128, 128)),
                               torchvision.transforms.ToTensor()])

  
        train_set_raw = torchvision.datasets.CelebA(root='../data', split='train', transform=transform)
        train_set = torch.utils.data.Subset(train_set_raw, list(range(0, len(train_set_raw), 5)))
        test_loader = DataLoader(torchvision.datasets.CelebA(root='../data', split='test', transform=transform))

        network = models.vgg16(pretrained=True).to(device)

    # generate random rotation matrix
    plain_size, param_size = get_nn_params(network)
    DIAG = random_diag(param_size)
    SENSINF = QUANTIZE_LEVEL + 1
    SENS1 = np.sqrt(plain_size) * CLIP_BOUND / Q + np.sqrt(2 * np.sqrt(plain_size) * CLIP_BOUND * np.log(2 / args.delta) / Q) + 4 * np.log(2 / args.delta) / 3
    SENS2 = CLIP_BOUND / Q + np.sqrt(SENS1 + np.sqrt(2 * np.sqrt(plain_size) * CLIP_BOUND * np.log(2 / args.delta) / Q))
    M = int(CYLIC_BOUND / np.log(PERROUND) - QUANTIZE_LEVEL)
    EPS_ = SENS2 * np.sqrt(2 * np.log(1.25/args.delta)) / S / np.sqrt(M*P*(1-P)) +(SENS2 * 5 * np.sqrt(np.log(10/args.delta)) / 2 + SENS1 / 3) / S / M / P / (1-P) / (1-args.delta/10)  + (2 * SENSINF * np.log(1.25/args.delta) / 3 + 2 * SENSINF * np.log(20*plain_size/args.delta) * np.log(10/args.delta) / 3) / S / M / P / (1-P)
    EPS = np.log(1+SUBSAMPLING_RATE * (np.exp(EPS_)-1))


    # Split into multiple training set
    TRAIN_SIZE = len(train_set) // NWORKER
    assert TRAIN_SIZE > 0, "Each worker should have at least one data point!"
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

    # define performance metrics
    ups = 0

    # analytical moments accountant
    if DP == 'dis-gauss':
        acct = rdp_acct.anaRDPacct()
        func_gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': np.sqrt(SIGMA2) / SENSITIVITY}, x)

    results = {'privacy':[], 'accuracy':[]}

    for epoch in tqdm(range(EPOCH)):
        # select workers per subset 
        print("Epoch: ", epoch)
        choices = np.random.choice(NWORKER, PERROUND)
        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
           params_copy.append(p.clone())
        params_flat_copy = flatten_params(list(network.parameters()), param_size)

        f_list = []
        for c in choices:
            # print(c)
            local_network = copy.deepcopy(network).to(device)
            if args.dataset == 'SVHN':
                optimizer = optim.SGD(local_network.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
            else:
                optimizer = optim.Adam(local_network.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            f_list.append(functools.partial(train, device, train_loaders[c], local_network, optimizer, criterion, DP, LOCALITER, params_flat_copy, param_size, CLIP_BOUND, SIGMA2, PERROUND, CYLIC_LEVEL, CYLIC_BOUND, M, P, INTERVAL, DEBUG, DIAG, QUANTIZE_LEVEL, QUANTIZE_BOUND, dataset=args.dataset))

        with Pool() as pool:
            local_grads = pool.map(smap, f_list)

        del f_list
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # aggregation
        average_grad = np.zeros(param_size)
        for local_grad in local_grads:
            average_grad = average_grad + local_grad / PERROUND

        params = list(network.parameters())
        if DEBUG == 'n' and DP != 'non-private':
            # print('here')
            average_grad = rotate(average_grad, DIAG, reverse=True)
        average_grad = recon_params(average_grad, network)
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)
        params_flat_copy = flatten_params(list(network.parameters()), param_size)
        flat2 = flatten_params(params, param_size)

        if DP == 'dis-gauss':
            acct.compose_subsampled_mechanism(func_gaussian, SUBSAMPLING_RATE)

        if (epoch+1) % CHECK_POINT == 0:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    if args.dataset == 'celebA':
                        target = target[:, 20].type(torch.long).to(device)
                    else:
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

        if (epoch+1) % 10 == 0:
            if DP == 'dis-gauss':
                output = open(RESULT_DIS_TEMPLATE%(DATASET, DP, args.dist, QUANTIZE_LEVEL, NBIT, SIGMA2, args.clipbound, args.nworker, args.perround), 'wb')
            elif DP == 'binom':
                output = open(RESULT_BINOM_TEMPLATE%(DATASET, DP, args.dist, QUANTIZE_LEVEL, NBIT, args.clipbound, args.nworker, args.perround), 'wb')
            else:
                output = open(RESULT_BASELINE_TEMPLATE%(DATASET, DP, args.dist, args.nworker, args.perround), 'wb')
            pkl.dump(results, output)
            output.close()
