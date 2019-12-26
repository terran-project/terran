.. _usage/quickstart:

Quickstart
==========

Once you have Terran installed (see :ref:`usage/installation`), you should be
ready to use it.

There are three main points that you should know about in order to use terran:

* The I/O helper functions that allow for reading and writing media.
* The actual models present in Terran, allowing you to perform face detection,
  face recognition and pose detection.
* The visualization utilities that can be used to, well, visualize the results.

Let's go over each of these.

I/O helper functions
--------------------

For videos, ``open_video``, ``write_video``. For images, ``open_image``,
``resolve_images``.

Available algorithms
--------------------

``face_detection``, ``extract_features``, ``pose_estimation``.

These functions are actually instances of three classes: ``Detection``,
``Recognition``, ``Estimation``. (Example of the default instantiation, why
would I want a custom one.)

Visualization utilities
-----------------------

``vis_poses``, ``vis_faces``

``display_image``
