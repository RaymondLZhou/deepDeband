import os
import shutil

from PIL import Image


def deband_image(file, gpu_ids):
    shutil.copyfile(f'temp/deepDeband-f/padded/{file}', f'temp/deepDeband-f/loaded/test/{file}')
    img = Image.open(f'temp/deepDeband-f/loaded/test/{file}')
    dim = max(img.size)
    img.close()

    os.chdir('../pytorch-CycleGAN-and-pix2pix')
    command = f'python test.py --name deepDeband-f --model test --netG unet_256 --norm batch \
        --dataroot ../src/temp/deepDeband-f/loaded --results_dir ../src/temp/deepDeband-f/debanded \
        --dataset_mode single --gpu_ids {gpu_ids} --preprocess none --crop_size {dim} --load_size {dim}'

    os.system(command)
    os.chdir('../src')


def process_image(file, image_size):
    img = Image.open(f'temp/deepDeband-f/debanded/deepDeband-f/test_latest/images/{file}')
    img = img.crop((0, 0, image_size[0], image_size[1]))
    img.save(f'../output/deepDeband-f/{file[:-9]+".png"}')


def deband_images(image_sizes, gpu_ids):
    for file in os.listdir('temp/deepDeband-f/padded'):
        for loaded_file in os.listdir('temp/deepDeband-f/loaded/test'):
            os.remove(f'temp/deepDeband-f/loaded/test/{loaded_file}')

        deband_image(file, gpu_ids)

    for file in os.listdir('temp/deepDeband-f/debanded/deepDeband-f/test_latest/images'):
        if file.endswith('_fake.png'):
            process_image(file, image_sizes[file[:-9]+'.png'])
