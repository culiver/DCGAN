import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        self.parser.add_argument("--mode",
                            type=str,
                            choices=['train', 'inference', 'valid'],
                            default='train',
                            help="operation mode")
        self.parser.add_argument("--weights_path",
                            type=str,
                            default='weights/{}.pkl'.format(full_weight_name),
                            help="model path for inference")
        self.parser.add_argument("--n_epochs",
                            type=int,
                            default=1000,
                            help="number of epochs of training")
        self.parser.add_argument("--batch_size",
                            type=int,
                            default=128,
                            help="size of the batches")
        self.parser.add_argument("--lr",
                            type=float,
                            default=0.0002,
                            # default=0.00005,
                            help="adam: learning rate")
        self.parser.add_argument("--b1",
                            type=float,
                            default=0.5,
                            # default=0.0,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2",
                            type=float,
                            default=0.999,
                            # default=0.9,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--latent_dim",
                            type=int,
                            default=100,
                            help="dimensionality of the latent space")
        self.parser.add_argument("--sample_interval",
                            type=int,
                            default=400,
                            help="interval between image sampling")
        self.parser.add_argument("--inference_num",
                            type=int,
                            default=1000,
                            help="number of generated images for inference")
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt