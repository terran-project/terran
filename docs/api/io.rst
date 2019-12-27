.. _api/io:

I/O functions
=============

Note that all functions within ``terran.io.video.*`` and ``terran.io.image.*`` are re-exported at
the level of ``terran.io.*``, and that's the recommended place to import them from.

.. _api-io-video:

terran.io.video
---------------

There are two functions related to video I/O, and are the main entry points for this purpose.

.. automodule:: terran.io.video
  :members: open_video, write_video

These functions are, however, just simple wrappers around the following two classes, which contain
the actual logic of reading and writing videos.

.. autoclass:: terran.io.video.reader.Video

.. autoclass:: terran.io.video.writer.VideoWriter
  :members:

.. _api-io-image:

terran.io.image
---------------

.. automodule:: terran.io.image
  :members: open_image, resolve_images
