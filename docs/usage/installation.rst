.. _usage/installation:

Installation
============

Before you start
----------------

Pytorch
^^^^^^^

Terran will install the default `Pytorch <https://pytorch.org/>`_ release in
during setup. If you need a custom option, such as supporting a different CUDA
version, or the non-GPU version, you'll need to install it yourself beforehand.

FFmpeg
^^^^^^

Terran makes use of `FFmpeg <https://www.ffmpeg.org>`_ in order to provide
I/O-related functions for videos. As such, make sure you have it installed as a
system dependency if you want to run predictions on videos.

OpenCV
^^^^^^

Terran will use `OpenCV <https://docs.opencv.org/master/>` to efficiently resize images and
considerably accelerate the face tracking process. Make sure you have all system dependencies to
install OpenCV correctly on your machine.

Cairo
^^^^^

By default, Terran will use `Pillow <https://pillow.readthedocs.io/>`_ to draw visualizations, but
it can also can also leverage `Cairo <https://www.cairographics.org/>`_ when present, making the
graphics better-looking. You can either install the ``pycairo`` package yourself or explicitly
install the dependency with::

  pip install terran[cairo]

feh
^^^

Terran can use `feh <https://feh.finalrewind.org/>`_ as backend to display
images (through the ``terran.vis.display_image`` function). If you don't have
it installed, it will use ``matplotlib`` as fallback, but we recommend the
former.


Installing from PyPI
--------------------

Terran may be installing by using ``pip`` with the following command::

  pip install terran


Installing from source
----------------------

Start by cloning the Terran repository::

  git clone https://github.com/nagitsu/terran.git

Then install the library by running::

  cd terran
  pip install -e .
