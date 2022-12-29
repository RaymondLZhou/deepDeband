import os
import shutil

import numpy as np
from PIL import Image


def deband_image(file, gpu_ids):
    img = Image.open(f'temp/deepDeband-w/padded/{file}')
    width, height = img.size

    for i in range(0, width-128, 128):
        for j in range(0, height-128, 128):
            patch = img.crop((i, j, i+256, j+256))
            patch.save(f'temp/deepDeband-w/loaded/test/{i}_{j}.png')

    img.close()
    num = len(os.listdir('temp/deepDeband-w/loaded/test'))

    os.chdir('../pytorch-CycleGAN-and-pix2pix')
    command = f'python test.py --name deepDeband-w --model test --netG unet_256 --norm batch \
        --dataroot ../src/temp/deepDeband-w/loaded --results_dir ../src/temp/deepDeband-w/debanded \
        --dataset_mode single --gpu_ids {gpu_ids} --crop_size 256 --load_size 256 --num_test {num}'

    os.system(command)
    os.chdir('../src')


def update_pixel_one(pixel_pos, pixels_data, pixels_new, dimensions):
    i, j = pixel_pos
    width, height = dimensions
    centre = (min(int(i/128)*128, width-256), min(int(j/128)*128, height-256))

    pixels_new[i, j] = pixels_data[centre][i-centre[0], j-centre[1]]


def update_pixel_two(pixel_pos, direction, pixels_data, image_centres, pixels_new, dimensions):
    i, j = pixel_pos
    width, height = dimensions

    if direction == 'lr':
        smaller = ((int(i/128)-1)*128, min(int(j/128)*128, height-256))
        larger = (int(i/128)*128, min(int(j/128)*128, height-256))
    elif direction == 'ud':
        smaller = (min(int(i/128)*128, width-256), (int(j/128)-1)*128)
        larger = (min(int(i/128)*128, width-256), int(j/128)*128)

    cur = np.array((i, j))

    if np.array_equal(cur, image_centres[larger]):
        pixels_new[i, j] = pixels_data[larger][i-larger[0], j-larger[1]]
        return

    w_smaller = 1 / np.linalg.norm(cur-image_centres[smaller])
    w_larger = 1 / np.linalg.norm(cur-image_centres[larger])

    weighted_smaller = [w_smaller*pixel for pixel in pixels_data[smaller][i-smaller[0], j-smaller[1]]]
    weighted_larger = [w_larger*pixel for pixel in pixels_data[larger][i-larger[0], j-larger[1]]]

    pixels = [left_pix+right_pix for (left_pix, right_pix) in zip(weighted_smaller, weighted_larger)]

    pixels_new[i, j] = tuple(int(pixel / (w_smaller+w_larger)) for pixel in pixels)


def update_pixel_four(pixel_pos, pixels_data, image_centres, pixels_new):
    i, j = pixel_pos

    top_left = ((int(i/128)-1)*128, (int(j/128)-1)*128)
    top_right = (int(i/128)*128, (int(j/128)-1)*128)
    bot_left = ((int(i/128)-1)*128, int(j/128)*128)
    bot_right = (int(i/128)*128, int(j/128)*128)

    cur = np.array((i, j))

    if np.array_equal(cur, image_centres[bot_right]):
        pixels_new[i, j] = pixels_data[bot_right][i-bot_right[0], j-bot_right[1]]
        return

    w_tl = 1 / np.linalg.norm(cur-image_centres[top_left])
    w_tr = 1 / np.linalg.norm(cur-image_centres[top_right])
    w_bl = 1 / np.linalg.norm(cur-image_centres[bot_left])
    w_br = 1 / np.linalg.norm(cur-image_centres[bot_right])

    weighted_tl= [w_tl*pixel for pixel in pixels_data[top_left][i-top_left[0], j-top_left[1]]]
    weighted_tr = [w_tr*pixel for pixel in pixels_data[top_right][i-top_right[0], j-top_right[1]]]
    weighted_bl = [w_bl*pixel for pixel in pixels_data[bot_left][i-bot_left[0], j-bot_left[1]]]
    weighted_br = [w_br*pixel for pixel in pixels_data[bot_right][i-bot_right[0], j-bot_right[1]]]

    pixels = [tl+tr+bl+br for (tl, tr, bl, br) in zip(weighted_tl, weighted_tr, weighted_bl, weighted_br)]

    pixels_new[i, j] = tuple(int(pixel / (w_tl+w_tr+w_bl+w_br)) for pixel in pixels)


def process_image(file, image_size):
    pixels_data = {}
    image_centres = {}

    padded_img = Image.open(f'temp/deepDeband-w/padded/{file}')
    width, height = padded_img.size
    padded_img.close()

    for i in range(0, width-128, 128):
        for j in range(0, height-128, 128):
            img = Image.open(f'temp/deepDeband-w/debanded/deepDeband-w/test_latest/images/{i}_{j}_fake.png')

            pixels_data[(i, j)] = img.load()
            image_centres[(i, j)] = np.array((i+127, j+127))

    img_new = Image.new("RGB", (width, height))
    pixels_new = img_new.load()

    for i in range(img_new.size[0]):
        for j in range(img_new.size[1]):
            if (i <= 127 or i >= width-128) and (j <= 127 or j >= height-128):
                update_pixel_one((i, j), pixels_data, pixels_new, (width, height))

            elif (127 <= i <= width-129) and (j <= 127 or j >= height-128):
                update_pixel_two((i, j), 'lr', pixels_data, image_centres, pixels_new, (width, height))

            elif (i <= 127 or i >= width-128) and (127 <= j <= height-129):
                update_pixel_two((i, j), 'ud', pixels_data, image_centres, pixels_new, (width, height))

            else:
                update_pixel_four((i, j), pixels_data, image_centres, pixels_new)

    img_new = img_new.crop((0, 0, image_size[0], image_size[1]))
    img_new.save(f'../output/deepDeband-w/{file}')


def deband_images(image_sizes, gpu_ids):
    for file in os.listdir('temp/deepDeband-w/padded'):
        shutil.rmtree('temp/deepDeband-w/debanded/deepDeband-w', ignore_errors=True)

        for loaded_file in os.listdir('temp/deepDeband-w/loaded/test'):
            os.remove(f'temp/deepDeband-w/loaded/test/{loaded_file}')

        deband_image(file, gpu_ids)
        process_image(file, image_sizes[file])
