import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = './data'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr = 1e-4   # instead of 1e-5
        self.batch_size = 2  # RTX 2050 should handle it
        self.epochs = 200
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.writer       = SummaryWriter()     # tensorboard writer

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)