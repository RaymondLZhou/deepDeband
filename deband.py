import deband_full
import deband_weighted


def deband_images(image_sizes, version, gpu_ids):
    if version == 'f':
        deband_full.deband_images(image_sizes, gpu_ids)
    elif version == 'w':
        deband_weighted.deband_images(image_sizes, gpu_ids)
