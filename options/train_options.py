from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument("--weights_path", type=str, default='weights/{}.pkl'.format( ull_weight_name),
                            help="model path for inference")
        self.parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
        self.parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, # default=0.00005,
                            help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, # default=0.0,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, # default=0.9,
                            help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        self.parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
        self.parser.add_argument("--inference_num", type=int, default=1000, help="number of generated images for inference")



        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = True