import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Compressing')
    parser.add_argument('--dataset', required=True, help='dataset path')
    parser.add_argument('--bit', required=True,
                        type=int, help='compression bit')
    parser.add_argument('--name', required=True, help='training name')

    parser.add_argument('--epoch_limit', type=int, default=0,
                        help='current run limitation (0 for no limit)')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count')
    parser.add_argument('--nepoch', type=int, default=50, help='# of epoch')

    # Training

    parser.add_argument('--warm', action='store_true',
                        help='warming up the training by first train encoder to atleast generate "similiar" image to input')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--debug', action='store_true', help='use debug mode?')
    parser.add_argument('--noscale', action='store_true',
                        help='use scale and random crop?')

    parser.add_argument('--epochsave', type=int, default=50, help='test')

    parser.add_argument('--batch_size', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=1, help='testing batch size')

    # Learning parameters
    parser.add_argument('--a', type=float, default=.8,
                        help='initial alpha gate for encoder')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use')

    parser.add_argument('--n_blocks', type=int, help='')
    parser.add_argument('--n_feature', type=int, help='')
    parser.add_argument('--padding', type=str, help='')
    parser.add_argument('--normalization', type=str, help='')
    parser.add_argument('--activation', type=str, help='')

    # Tensorboard parameters
    parser.add_argument('--commit', action='store_true',
                        default=False, help='commit mode? checkpoint will replace and save all models configuration for retraining')
    parser.add_argument('--tensorboard', default=True, action='store_true',
                        help='use tensorboard?')
    parser.add_argument(
        '--hookbin', help='hookbin url for capturing tensorboard url')
    parser.add_argument(
        '--auth_token', help='auth token for ngrok for limits')

    parser.add_argument('--silent', action='store_true',
                        help='silent the tqdm output')
    parser.add_argument('--optimized_encoder', action='store_true',
                        help='using new encoder logic?')

    opt = parser.parse_args()

    return opt
