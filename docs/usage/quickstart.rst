.. _usage/quickstart:

Getting started
===============

Once you have Terran installed (see :ref:`usage/installation`), you should be ready to use it. This
section offers an overview of what's possible and what tools are available within the library.

There are three aspects of Terran you should be familiar with in order to use it:

* The **I/O helper functions** that allow for reading and writing media.
* The actual **models** present in Terran, allowing you to perform face detection, face recognition
  and pose detection.
* The **visualization utilities** that can be used to, well, visualize the results.

Let's now go over each of these.

Reading and writing images and videos
-------------------------------------

Terran provides utilities for reading and writing different media through a simple interface, so
you don't have to install extra libraries or write dozens of lines of code just to read frames
efficiently out of a video.

First of all is, of course, reading images (see :ref:`api-io-image`). The function
:func:`open_image <terran.io.image.open_image>` is a simple wrapper around ``PIL.Image.open``  that
returns a ``numpy.ndarray`` of size ``(height, width, channels)`` representing the image. All image
data within Terran is specified in the :math:`HxWxC` format.

The function also allows reading remote images, so you could do as follows::

    # All I/O functions are exposed at the ``terran.io`` level too.
    from terran.io import open_image

    image = open_image(
        'https://raw.githubusercontent.com/nagitsu/terran'
        '/master/examples/readme/many-faces-raw.jpg'
    )

    print(image.shape)
    # >> (1280, 1920, 3).

For reading videos, you can use :func:`open_video <terran.io.video.open_video>`. This function
opens a resource containing a video and returns a :class:`Video <terran.io.video.Video>` class
representing it. In order to obtain the actual frames, the class may be iterated on, as it behaves
like a generator. Frames will be yielded one by one, or by batches, depending on the ``batch_size``
parameter passed to the function.

:func:`open_video <terran.io.video.open_video>` can handle several sources of video:

* Local video files, such as ``short_movie.mkv`` or ``dog.mp4``.
* URLs to a video stream, such as ``http://127.0.0.1:8000/my-stream/``, if you were streaming
  something locally. This is any stream that can be read by ``FFmpeg``.
* Path to the webcam devices, such as ``/dev/video0``. This will open the resource and start
  streaming from it, if it's available.
* Videos hosted on video platforms supported by ``Youtube-DL``. This means a URL pointing to a e.g.
  Youtube or Vimeo video.

As an example, this would open and start yielding frames from a Youtube video::

    from terran.io import open_video

    video = open_video(
      'https://www.youtube.com/watch?v=oHg5SJYRHA0',
      start_time='00:00:30',
      read_for=30,
      batch_size=32,
    )

    for frames in video:
        print(frames.shape)
        # >> (32, 1280, 1920, 3)

As you can see above, you can select the starting time, the time to read for, the batch size and
more. See the function reference for all the available options.

You can also write videos through the :func:`write_video <terran.io.video.write_video>` function.
Calling this function and pointing it to a path will return a :class:`VideoWriter
<terran.io.video.writer.VideoWriter>` object and create a video file, exposing a :func:`write_frame
<terran.io.video.writer.VideoWriter.write_frame>` method that receives an image and adds it to the
video. For instance::

    from terran.io import write_video

    writer = write_video('dog.mp4')

    # `images` is an array of images of the same size.
    for image in images:
        writer.write_frame(image)

    # We *must* close the writer, or the video file will be corrupt.
    writer.close()

Both the :class:`Video <terran.io.video.Video>` and :class:`VideoWriter
<terran.io.video.VideoWriter>` classes will perform the reading and writing through ``FFmpeg`` in a
background thread, in order to avoid blocking the program while video is read and memory is copied
over. This improves resource utilization by quite a lot.

.. You can read more at :ref:`usage/io`.

These are not all the I/O functions available, and not all they can do; you can check :ref:`api/io`
for more information.

Available algorithms
--------------------

But, of course, we're not here for the I/O functions.

``face_detection``, ``extract_features``, ``pose_estimation``.

These functions are actually instances of three classes: ``Detection``,
``Recognition``, ``Estimation``. (Example of the default instantiation, why
would I want a custom one; point to usage/algorithms, don't explain)

checkpoints, prompted to that, maybe terran checkpoint utility.

Visualization utilities
-----------------------

``vis_poses``, ``vis_faces``

``display_image``
