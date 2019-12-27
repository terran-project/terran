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

Interacting with people
-----------------------

But, of course, we're not here for the I/O functions. Let's see how Terran can help us locate and
interact with people in images and videos.

Detecting faces
^^^^^^^^^^^^^^^

Given an image or a batch of images (say, the batched frames returned by iterating over a
:class:`Video <terran.io.video.Video>` instance), you can call :func:`face_detection
<terran.face.detection.face_detection>` to obtain the faces present on them.

For each image, Terran will return a list of faces found. Each face is represented by a dictionary
containing three keys:

* ``bbox`` which is a ``numpy.ndarray`` of size ``(4,)``, containing the coordinates of the
  bounding box that surrounds the face. These coordinates are in a :math:`(x_{min}, y_{min},
  x_{max}, y_{max})` format.

* ``landmarks`` which is a ``numpy.ndarray`` of size ``(5, 2)``, containing the :math:`(x, y)`
  coordinates of five facial landmarks of the face. This can be (and is) used by downstream
  algorithms to align the face correctly before processing.

* ``score`` which is a ``numpy.ndarray`` of size ``(1,)``, with the confidence score of the
  detected face, a value between 0 and 1.

Terran does its best to match the return type to whichever input was sent into. This means that if
you, for instance, send in a single image, you'll receive a single list containing each face data.
If you, however, send in a batch of images, the function will return a list containing a list of
faces for each image.

Imagine we have the image we loaded on the previous section using :func:`open_image
<terran.io.image.open_image>`. We can detect all of the faces present by passing it to
:func:`face_detection <terran.face.detection.face_detection>`::

    print(image.shape)
    # >> (1280, 1920, 3).

    # All face-related functions are re-exported at the `terran.face`
    # level.
    from terran.face import face_detection

    faces = face_detection(image)
    for face in faces:
        print(face)
        print('bbox = ({}); landmarks = ({}); conf = {:.2f}'.format(
            ', '.join(map(str, face['bbox'])),
            ' '.join(map(str, face['landmarks'])),
            face['score']
        ))

    # >> bbox = (1326, 1048, 1475, 1229); landmarks = ([1360 1115] [1427 1116] [1390 1156] [1367 1183] [1421 1183]); conf = 1.00
    # >> bbox = (590, 539, 690, 667); landmarks = ([604 583] [647 586] [615 612] [608 633] [642 635]); conf = 0.99
    # >> bbox = (1711, 408, 1812, 530); landmarks = ([1731  451] [1775  451] [1747  477] [1735
    499] [1769  499]); conf = 0.99

If you were to send a batch of frames, for instance, the return type would be different::

    print(frames.shape)
    # >> (32, 1280, 1920, 3)

    faces_per_frame = face_detection(frames)
    print(len(faces_per_frame))
    # >> 32
    print(type(faces_per_frame[0]))
    # >> list
    print(len(faces_per_frame[0]))
    # >> 1

Recognizing faces
^^^^^^^^^^^^^^^^^

The task of face recognition aims to give a unique representation to a face. In a perfect scenario,
this representation would be robust to changes in appearance, such as the person growing a beard or
changing their hairstyle. Of course, that's very difficult to achieve. What we try to do, instead,
is extract features out of the face, represented by a N-dimensional vector (a ``numpy.ndarray`` of
shape ``(N,)``) that is as stable as possible across appearence changes.

Through the function :func:`extract_features <terran.face.recognition.extract_features>`, you can
extract these features. If you run it through a face, such as the ones detected above, you'll get a
dense representation of it. This representation is constructed so that, if you take two faces of
the same person, the cosine distance between their features should be very small.

This is better illustrated with an example. Let's take the following three images:

(images stitched together)

We can obtain the representations of each as follows::

    from terran.face import extract_features

    # We'll go over on how exactly the function is called in a bit.
    features_rw1 = extract_features(
        rw1, faces_per_image=face_detection(rw1)
    )[0]
    features_rw2 = extract_features(
        rw2, faces_per_image=face_detection(rw2)
    )[0]
    features_th = extract_features(
        th, faces_per_image=face_detection(th)
    )[0]

    # In this case, the vector dimension, N, is 512:
    print(features_rw1.shape)
    # >> (512,)

    # We can compare the vectors using the cosine distance.
    from scipy.spatial.distance import cosine

    # If the distance between two faces is below 0.7, it's probably the
    # same person. If it's below 0.4, you can be almost certain it is.
    print(cosine(features_rw1, features_rw2))
    # >> 0.5384056568145752
    print(cosine(features_rw1, features_th))
    # >> 1.0747144743800163
    print(cosine(features_rw2, features_th))
    # >> 1.06807991117239

As you can see, extracting features on a face will give us a vector of shape ``(512,)`` that, along
with the ``cosine`` function, will help us identify a person across images.

The function :func:`extract_features <terran.face.recognition.extract_features>` can be called in
two ways:

* Like we did above, by sending in the image and passing the faces detected by
  :func:`face_detection <terran.face.detection.face_detection>` in the ``faces_per_image`` optional
  parameter. This will make the function return a list with one entry per image and, within each,
  a list of one entry per face containing the features. (Note that this is why we used the ``[0]``,
  to obtain the features for the first -and only- face.)
* By sending in a list of already-cropped faces. You just send in a list of faces and you receive a
  list of features.

See :ref:`usage/algorithms` for more information into why these alternatives exist and what's the
recommended way of calling it (hint: it's the first one!).

Estimating poses of people
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use :func:`pose_estimation <terran.pose.pose_estimation>` to obtain the poses of the people
present in an image. The process is similar to how we did it with face detection: you pass in an
image, and you obtain the coordinates of each keypoint of the pose.

In this case, instead of the ``bbox`` and ``landmarks`` keys, you'll get a ``keypoints`` key
containing a ``numpy.ndarray`` of shape ``(18, 3)``, consisting of the 18 keypoints detected for
poses::

    image = open_image(
        'https://raw.githubusercontent.com/nagitsu/terran'
        '/master/examples/readme/many-poses-raw.jpg'
    )

    from terran.pose import pose_estimation

    poses = pose_estimation(image)
    print(len(poses))
    # >> 6

    print(poses[0]['keypoints'])
    # >> array([[  0,   0,   0],
    # >>        [714, 351,   1],
    # >>        ...
    # >>        [  0,   0,   0],
    # >>        [725, 286,   1],
    # >>        [678, 292,   1]], dtype=int32)

The ``keypoints`` array has three columns: the first two are the :math:`(x, y)` coordinates, while
the third is either 0 or 1, indicating whether the keypoint is visible or not.

Additional notes
^^^^^^^^^^^^^^^^

checkpoints, prompted to that, maybe terran checkpoint utility.

Note that for every algorithm, Terran provides a function-based version that allows minimal
customization (such as :func:`face_detection <terran.face.detection.face_detection>`) and a
class-based version that can be configured as needed, allowing you to change detection thresholds,
internal image resizing, batching, and such. The functions above are actually instantiations of
these classes with default settings. We only touched upon the shortcut functions-based versions
here, so be sure to check :ref:`usage/algorithms` for more information.


Visualizing the results
-----------------------

``vis_poses``, ``vis_faces``

``display_image``
