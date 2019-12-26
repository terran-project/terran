.. _api/models:

I/O functions
=============

Video
-----

There are two functions related to video I/O, and are the main entry points for
this purpose.

.. automodule:: terran.io.video
  :members: open_video, write_video

These functions are, however, just simple wrappers around the following two
classes, which contain the actual logic of reading and writing videos.

.. autoclass:: terran.io.video.reader.Video

.. autoclass:: terran.io.video.writer.VideoWriter
  :members:


Images
------

.. automodule:: terran.io.image
  :members: resolve_images, open_image
