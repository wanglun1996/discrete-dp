import numpy as np
import argparse
import pickle as pkl
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULT_TEMPLATE = '../results/pkl/%s-%s-%d-%d-%f.pkl'
PDF_TEMPLATE = '../results/pdf/%s-%s-%d-%d-%f.pdf'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--dp', default='dis-gauss')
    parser.add_argument('--nbit', type=int, default=20)
    parser.add_argument('--quanlevel', type=int, default=201)
    parser.add_argument('--param', type=float, default=0.25)
    args = parser.parse_args()

    pkl_file = open(RESULT_TEMPLATE%(args.dataset, args.dp, args.quanlevel, args.nbit, args.param), 'rb')
    mydict = pkl.load(pkl_file)
    pkl_file.close()

    privacy = mydict['privacy']
    accuracy = np.array([x.double() for x in mydict['accuracy']])
    print(privacy)
    print(accuracy)

    ax = plt.subplot()
    ax.set_xscale("log", nonposx='clip')

    ax.errorbar(privacy, accuracy, capsize=5, capthick=2, elinewidth=1, linestyle='dashed', marker='o')

    # plt.xlabel('Epsilon', fontsize=20)
    # plt.ylabel(args.metric, fontsize=20)
    # plt.legend()
    # plt.grid(True)

    with PdfPages(PDF_TEMPLATE%(args.dataset, args.dp, args.quanlevel, args.nbit, args.param)) as pdf:
        pdf.savefig(bbox_inches='tight')
