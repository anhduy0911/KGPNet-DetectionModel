
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    pass


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pill Detection')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=911, metavar='S',
                        help='random seed (default: 911)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', dest='batch_size')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', dest='v_batch_size')
    parser.add_argument('--name', type=str, default="baseline", metavar='N',
                        help='name of saving model')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='patience of early stopping')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--backbone', type=str, default="resnet50", metavar='N', help='choose backbone model')
    parser.add_argument('--mode', type=str, default="train", metavar='N', help='train or test')

    args = parser.parse_args()
    
    # init_distributed_mode(args)
    # seed_everything(args.seed)
    main(args)