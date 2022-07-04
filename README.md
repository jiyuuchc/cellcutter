# cellcutter
Cellcutter is a simple library for cell segmentation from microscopy images. It is designed to work on crowded cell populations.

Note: Even though _cellcutter_ uses a neural network, it is _not_ a deep-learning model in the traditional sense. It is intended to be used directly on the image you want to segment, as if it is a classical unsupervised segmentation algorithm (e.g. watershed). If you are more interesed in building a generalizable deep-learning model, you should check out our project [LACSS](https://github.com/jiyuuchc/lacss/).

Easiest way to see how _cellcutter_ works it to check the notebook under the notebooks folder. The demo notebook can be run in google colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/cellcutter/blob/main/notebooks/demo.ipynb)
