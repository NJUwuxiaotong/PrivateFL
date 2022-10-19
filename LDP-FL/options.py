"""Parser options."""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(
        description='Reconstruct some image from a trained model in '
                    'federated learning environment.')

    # client information:
    parser.add_argument("--client_no", default=100, type=int,
                        help='the number of clients')
    parser.add_argument("--client_ratio", default=0.1, type=float,
                        help='the ratio of clients of global model update')
    parser.add_argument("--is_iid", default=True, type=bool,
                        help='whether the dataset is dispatched iid')

    # dataset information
    parser.add_argument('--dataset', default='mnist', type=str)

    # model training information
    parser.add_argument('--model_name', default='mnist mlp', type=str,
                        help='Vision model.')
    parser.add_argument("--round_no", default=1500, type=int,
                        help='the number of round of global model')
    parser.add_argument('--epoch_no', default=5, type=int,
                        help='If using a trained model, how many epochs was '
                             'it trained?')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='the learning ratio in model training')

    # invert gradient attack
    parser.add_argument('--attack_app', default=None, type=str,
                        help='attack approach')
    parser.add_argument('--attack_no', default=None, type=int,
                        help='The numbe of attacks')
    parser.add_argument('--demo_target', default=False, type=bool,
                        help='Cifar validation image used for reconstruction.')
    parser.add_argument('--dtype', default='float', type=str,
                        help='Data type used during reconstruction '
                             '[Not during training!].')
    parser.add_argument('--trained_model', action='store_true',
                        help='Use a trained model.')
    parser.add_argument('--accumulation', default=1, type=int,
                        help='Accumulation 0 is rec. from gradient, '
                             'accumulation > 0 is reconstruction '
                             'from fed. averaging.')
    parser.add_argument('--num_images', default=1, type=int,
                        help='How many images should be recovered from the '
                             'given gradient.')
    parser.add_argument('--label_flip', action='store_true',
                        help='Dishonest server permuting weights '
                             'in classification layer.')

    # Rec. parameters
    parser.add_argument('--optim', default='ours', type=str,
                        help='Use our reconstruction method or the DLG method.')
    parser.add_argument('--restarts', default=1, type=int,
                        help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str,
                        help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str,
                        help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str,
                        help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false',
                        help='Do not used signed gradients.')
    parser.add_argument('--boxed', action='store_false',
                        help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str,
                        help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str,
                        help='Choice of image initialization.')
    parser.add_argument('--tv', default=1e-4, type=float,
                        help='Weight of TV penalty.')

    # Files and folders:
    parser.add_argument('--save_image', default=True, action='store_true',
                        help='Save the output to a file.')
    parser.add_argument('--image_path', default='attack_results/images/',
                        type=str, help='the path to save the reconstruction '
                                       'image')
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str,
                        help='Name tag for the result table and model.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true',
                        help='Run everything for just one step to '
                             'test functionality.')
    return parser
