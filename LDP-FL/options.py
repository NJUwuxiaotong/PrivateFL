"""Parser options."""

import argparse

def options():
    """Construct the argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(
        description='Privacy attack and preservation in federated learning.')

    # federated learning:
    parser.add_argument('--model', default='ConvNet', type=str,
                        help='Neural network model.')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='Data set.')
    parser.add_argument('--client_no', default=100, type=int,
                        help='The number of clients.')
    parser.add_argument('--client_ratio', default=0.2, type=float,
                        help='The ratio of clients for federated learning.')
    parser.add_argument('--is_iid', default=True, type=bool,
                        help='Whether data distribution of a client is iid.')
    parser.add_argument('--round_no', default=500, type=int,
                        help='The number of training rounds.')
    parser.add_argument('--epochs', default=120, type=int,
                        help='If using a trained model, '
                             'how many epochs was it trained?')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rating')

    parser.add_argument('--accumulation', default=0, type=int,
                        help='Accumulation 0 is rec. from gradient, '
                             'accumulation > 0 is reconstruction from '
                             'fed. averaging.')

    return parser
