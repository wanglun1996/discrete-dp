import numpy as np
import argparse
import pickle as pkl
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULT_DIS_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f-%f-%d-%d.pkl'
RESULT_BINOM_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f-%d-%d.pkl'
RESULT_BASELINE_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d.pkl'
# RESULT_TEMPLATE = '../results/pkl/%s-%s-%s-%d-%d-%f.pkl'
PDF_TEMPLATE = '../results/pdf/%s-%s-%s-%d-%d-%f.pdf'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--dp', default='dis-gauss')
    parser.add_argument('--dist', default='homo')
    parser.add_argument('--nbit', type=int, default=16)
    parser.add_argument('--quanlevel', type=int, default=201)
    parser.add_argument('--sigma2', type=float, default=0.36)
    parser.add_argument('--clipbound', type=float, default=0.25)
    parser.add_argument('--nworker', type=int, default=20)
    parser.add_argument('--perround', type=int, default=10)
    args = parser.parse_args()

    if args.dp == 'non-private':
        file_name = RESULT_BASELINE_TEMPLATE%(args.dataset, args.dp, args.dist, args.nworker, args.perround)
    elif args.dp == 'dis-gauss':
        file_name = RESULT_DIS_TEMPLATE%(args.dataset, args.dp, args.dist, args.quanlevel, args.nbit, args.sigma2, args.clipbound, args.nworker, args.perround)
    elif args.dp == 'binom':
        file_name = RESULT_BINOM_TEMPLATE%(args.dataset, args.dp, args.dist, args.quanlevel, args.nbit, args.clipbound, args.nworker, args.perround)

    pkl_file = open(file_name, 'rb')
    mydict = pkl.load(pkl_file)
    pkl_file.close()

    privacy = mydict['privacy']
    accuracy = np.array([x.double() for x in mydict['accuracy']])
    # print(accuracy)
    print("x,\ty")
    if args.dp == 'non-private':
        for idx, acc in enumerate(accuracy):
            print("%d,\t%f"%(idx, acc.item()))
    else:
        for priv, acc in zip(privacy, accuracy):
            print("%f,\t%f"%(priv, acc.item()))

    # ax = plt.subplot()
    # ax.set_xscale("log", nonposx='clip')
    # ax.errorbar(privacy, accuracy, capsize=5, capthick=2, elinewidth=1, linestyle='dashed', marker='o')
    # plt.xlabel('Epsilon', fontsize=20)
    # plt.ylabel(args.metric, fontsize=20)
    # plt.legend()
    # plt.grid(True)

    # with PdfPages(PDF_TEMPLATE%(args.dataset, args.dp, args.dist, args.quanlevel, args.nbit, args.param)) as pdf:
    #     pdf.savefig(bbox_inches='tight')
