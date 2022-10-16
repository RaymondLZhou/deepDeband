import os

from PIL import Image, ImageOps


def new_dimentions(width, height):
    new_width = (width // 256 + 1) * 256 if width % 256 != 0 else width
    new_height = (height // 256 + 1) * 256 if height % 256 != 0 else height

    return new_width, new_height


def pad_image(img_path, image_sizes):
    img = Image.open(img_path)
    width, height = img.size
    new_width, new_height = new_dimentions(width, height)
    image_sizes[img_path.split('/')[-1]] = (width, height)

    padded_img = Image.new("RGB", (max(2*width, new_width), max(2*height, new_height)))
    flipped_img = ImageOps.flip(img)
    mirrored_img = ImageOps.mirror(img)
    flipped_mirrored_img = ImageOps.flip(ImageOps.mirror(img))

    for i in range(0, new_width+1, width*2):
        for j in range(0, new_height+1, height*2):
            padded_img.paste(img, (i, j))
            padded_img.paste(flipped_img, (i, j+height))
            padded_img.paste(mirrored_img, (i+width, j))
            padded_img.paste(flipped_mirrored_img, (i+width, j+height))

    padded_img = padded_img.crop((0, 0, new_width, new_height))
    return padded_img


def pad_images(image_sizes, version):
    for file in os.listdir(f'temp/deepDeband-{version}/padded'):
        os.remove(f'temp/deepDeband-{version}/padded/{file}')

    for file in os.listdir('input'):
        padded_img = pad_image(f'input/{file}', image_sizes)
        padded_img.save(f'temp/deepDeband-{version}/padded/{file}')
