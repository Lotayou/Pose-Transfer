from .base_funit_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--G1_lr', type=float, default=0.0001, help='initial learning rate for optimizer')
        self.parser.add_argument('--G2_lr', type=float, default=0.0001, help='initial learning rate for optimizer')
        self.parser.add_argument('--D_lr', type=float, default=0.0001, help='initial learning rate for optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.00001, help='initial learning rate for optimizer')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_l1', type=float, default=10.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=5.0, help='weight for perceptual L1 loss')
        self.parser.add_argument('--lambda_gan', type=float, default=5.0, help='weight of GAN loss')

        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        
        self.parser.add_argument('--L1_type', type=str, default='origin', help='use which kind of L1 loss. (origin|l1_plus_perL1)')
        self.parser.add_argument('--perceptual_layers', type=int, default=3, help='index of vgg layer for extracting perceptual features.')
        self.parser.add_argument('--percep_is_l1', type=int, default=1, help='type of perceptual loss: l1 or l2')
        self.parser.add_argument('--no_dropout_D', action='store_true', help='no dropout for the discriminator')
        self.parser.add_argument('--DG_ratio', type=int, default=1, help='how many times for D training after training G once')

        self.parser.add_argument('--use_custom_vgg_weights', action='store_true')

        self.isTrain = True
        
        ###############
        # Obsolete
        ###############
        