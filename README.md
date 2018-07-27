# Generative-Temporal-Models-with-Spatial-Memory

This repo contains code accomplishing the paper, 	[Generative Temporal Models with Spatial Memory
for Partially Observed Environments (Marco Fraccaro et al.)](https://arxiv.org/abs/1804.09401). It includes code for running the image navigation experiment.


### Dependencies
This code requires the following:
* python 3.\*
* pytorch v 0.4.0
* pyflann v 1.6.14
* some other frequent used packages, numpy, matplotlib etc.


### Data
This experiment is traind by the Large-scale CelebFaces Attributes (CelebA) Dataset, downloaded from (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Moreover, in this experiment, the preprocess about face detector is required. So I save the preprocess result in `./datasets/CelebA.zip`. Readers can unzip this into `./datasets/` for further experiment.


### Usage
Instructions of python files.
- `config.py`
  Use for setting the parameters of the model, such as the batch_size, total epochs, log interval and so on.
- `main.py`
  It is **the main function** that uses to train our GTM-SM model. It calls for the functions -- `train` and `test` in `train.py` to train our model and feedback the reconstructon error from validation set.
- `train.py`
  Use for implementation of the `train` and `test` function.
- `roam.py`
  Use for genetating the trajectory of the 8 x 8 crop over a 32 x 32 image.
- `show_results.py`
  Use for generating the result as the `./videos/image_navigation` shows.
- `sample.py`
  Use for generating image navigation experiment videos. It can be directly called to producing the corresponding result.
- `/utils/torch_utils.py`
  Provide some useful functions.
  
I have saved the trained parameters in `./saves/`, and the file `sample.py` would reload these parameters to do the image navigation, so that it can reproduce the simulation result.

If readers would like to check the result, you can directly run the file `sample.py`. For convenience, it is pleasant to have seen it first, so I recode the relevant videos in `videos/image_navigation/`.


### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/chenxy99/Generative-Temporal-Models-with-Spatial-Memory/issues).
