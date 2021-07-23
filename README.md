# tensorflow-unet

![https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg)

[U-Net](https://arxiv.org/abs/1505.04597v1) (2015) implemented from scratch in Tensorflow and trained on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

[U-Net](https://arxiv.org/abs/1505.04597v1) is a convolutional neural network architecture composed of an encoder and decoder pipeline, the latter utilizing contextual information from the former in order to perform image segmentation (the classification of each pixel according to any number of labels).

In this repository, I've synthesized a number of online tutorials (cited in `main.py`) so as to implement a model from scratch in Tensorflow, which I've trained to distinguish 37 species of household cats and dogs (via the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (2012)) within ~4,000 images of varying resolution and aspect ratio, experimenting with methods of image mirroring (to extrapolate missing contextual information) and optimization (e.g. depthwise separable convolution, normalization, and horizontal training image flipping) along the way.

## Execution

`$ pipenv install`

`$ pipenv run ./main.py`

NOTE: Prior to training, it will attempt to load the weights of the model from `weights/cp.ckpt` (which is written to after every epoch).
