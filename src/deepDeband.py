import argparse

import cleanup
import deband
import padding

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepDeband')
    parser.add_argument('--version', required=True, choices=['f', 'w'], help='deepDeband version [f | w]')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0; 0,1,2; 0,2. use -1 for CPU')
    args = vars(parser.parse_args())

    image_sizes = {}

    cleanup.cleanup()
    cleanup.setup(args['version'])
    padding.pad_images(image_sizes, args['version'])
    deband.deband_images(image_sizes, args['version'], args['gpu_ids'])
    cleanup.cleanup()
