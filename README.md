# Deep Image Debanding
This repository contains the implementation for the following work:

Raymond Zhou, Shahrukh Athar, Zhongling Wang, and Zhou Wang, “Deep Image Debanding,” in IEEE International Conference on Image Processing (ICIP), Bordeaux, France, Oct. 2022, [IEEE Xplore](https://doi.org/10.1109/ICIP46576.2022.9897680)


## Abstract
Banding or false contour is an annoying visual artifact whose impact negatively degrades the perceptual quality of visual content. Since users are increasingly expecting better visual quality from such content and banding leads to deteriorated quality-of-experience, the area of banding removal or debanding has taken paramount importance. Existing debanding approaches are mostly knowledge-driven, while data-driven debanding approaches remain surprisingly missing. In this work, we construct a large-scale dataset of 51,490 pairs of corresponding pristine and banded image patches, which enables us to make one of the first attempts at developing a deep learning based banding artifact removal method for images that we name _deep debanding network_ (deepDeband). We also develop a bilateral weighting scheme that fuses patch-level debanding results to full-size images. Extensive performance evaluation shows that deepDeband is successful at greatly reducing banding artifacts in images, outperforming existing methods both quantitatively and visually.

## Banding Example
### Banded image
![banded](res/images/banded.png) 

### Debanded (deepDeband-f)
![debanded full](res/images/full.png)

### Debanded (deepDeband-w)
![debanded weighted](res/images/weighted.png)


## Running the Model
1. Install the prerequisites with ```pip install -r requirements.txt```
2. Place your input images in [input/](input/)
3. Navigate to the src directory with ```cd src```
4. Choose the version of deepDeband to run (f or w) with ```python deepDeband.py --version f``` or ```python deepDeband.py --version w```
5. The output images are located in [output/deepDeband-f/](output/deepDeband-f/) or [output/deepDeband-w/](output/deepDeband-w/)


## Dataset
The dataset of 51,490 pairs of matching banded and pristine image patches of size 256x256 is available here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7224906.svg)](https://doi.org/10.5281/zenodo.7224906)


## Model
In case there are any issues with downloading the model in this repo through GitHub LFS, it is also available here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523437.svg)](https://doi.org/10.5281/zenodo.7523437). After downloading and extracting the zip file, place the ```deepDeband-f``` and ```deepDeband-w``` folders into [pytorch-CycleGAN-and-pix2pix/checkpoints/](pytorch-CycleGAN-and-pix2pix/checkpoints/).
