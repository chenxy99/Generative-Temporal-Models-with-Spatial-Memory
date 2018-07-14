# Generative-Temporal-Models-with-Spatial-Memory

This repo contains code accompaning the paper, 	[Generative Temporal Models with Spatial Memory
for Partially Observed Environments (Marco Fraccaro et al.)](https://arxiv.org/abs/1804.09401). It includes code for running the image navigation experiment.

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* pytorch v0.4.0
* pyflann v1.6.14

### Data
For the Multi-Task Facial Landmark (MTFL) data, download from (http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip). Extract 2560  files from `./MTFL/AFLW` to `./datasets/MTFL_data/training` and extract other 432  files from `./MTFL/AFLW` to `./datasets/MTFL_data/testing`.

### Usage
Instructions of python files.
- `config.py`
  Use for setting the parameters of the model, such as the batch_size, total epochs, log interval and so on.
- `main.py`
  Use for trian our GTM-SM model, it calls for the functions -- `train` and `test` to train our model and feedback the reconstructon error from validation set.
- `train.py`
  Use for implementation of the `train` and `test` function.
- `Roam.py`
  Use for genetating the trajectory of the 8 x 8 crop.
- `show_results.py`
  Use for generating the result as the `videos/image_navigation` shows.
- `sample.py`
  Use for generating image navigation experiment figure. It can be directly called to producing the corresponding result.
- `/utils/torch_utils.py`
  Provide some usefull funtions.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/chenxy99/Generative-Temporal-Models-with-Spatial-Memory/issues).
