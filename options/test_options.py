from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes train options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, train, etc')
        # Dropout and Batchnorm has different behavioir during training and train.
        parser.add_argument('--eval', action='store_true', help='use eval mode during train time.')
        parser.add_argument('--num_test', type=int, default=5000, help='how many train images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
