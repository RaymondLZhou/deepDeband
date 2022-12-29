import os
import shutil


def setup(version):
    os.mkdir('temp')
    os.mkdir(f'temp/deepDeband-{version}')
    os.mkdir(f'temp/deepDeband-{version}/debanded')
    os.mkdir(f'temp/deepDeband-{version}/loaded')
    os.mkdir(f'temp/deepDeband-{version}/loaded/test')
    os.mkdir(f'temp/deepDeband-{version}/padded')


def cleanup():
    shutil.rmtree('temp', ignore_errors=True)
