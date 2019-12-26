terran

---

**Terran** is a human perception library that provides computer vision
techniques and algorithms in order to facilitate building systems that interact
with people.

The philosophy behind the library is to focus on tasks and problems instead of
models and algorithms. That is, we aim to always have the best possible
algorithm for the job given its constraints, and to take the burden of finding
which model performs best off you.

The library strives to be friendly and easy-to-use. It's written fully in
Python and Pytorch, avoiding C++ extensions as much as possible, in order to
avoid difficulties in installation. Just `pip install` and you're good to go!

We (currently) provide models for: **face detection**, **face recognition** and
**pose estimation**. We also offer several utility functions for efficiently
reading and visualizing results, which should simplify work a bit.

<p align="center">
  <img src="examples/readme/main-image.jpg", width="720"><br>
  <sup>
    Example of Terran's face detection and pose estimation capabilities.
  </sup>
</p>

# Features

* Efficient I/O utilities to read and write videos through `ffmpeg`. Frames are
  pre-fetched in a background thread, allowing you to maximize GPU usage when
  processing videos.

* Utilities to open remote images, recursively find images, and (prettily)
  visualize results.

* Checkpoint management tool, so you don't have to manually download
  pre-trained model files.

* Face detection provided through the *RetinaFace* model.

* Face recognition provided through the *ArcFace* model.

* Pose estimation provided through the *OpenPose* model (2017 version).

# Getting started

Be sure to read the full documentation [here]().

## Installation

Terran requires Python 3.6 or above, and Pytorch 1.3 or above. It can be used
with or without a GPU, though the current available algorithms require GPUs in
order to run under a reasonable time.

To install, run:

```bash
pip install terran
```

If you require a particular Pytorch version (e.g. you're using a specific CUDA
version), be sure to install it beforehand.

## Usage

See the [Examples](#examples) section for more in-depth examples.

You can use the functions under `terran.io.*` for easy reading of media files,
and the appropriate algorithm function under the top-level module. If you don't
need [any customization](#customizing-model-settings), just issue the following
in an interactive console:

```python
>>> from terran.io import open_image
>>> from terran.vis import display_image, vis_faces
>>> from terran.face import face_detection
>>>
>>> image = open_image('examples/readme/many-faces-raw.jpg')
>>> detections = face_detection(image)
>>> display_image(vis_faces(image, detections))
```

<p align="center">
  <img src="examples/readme/many-faces.jpg", width="720">
</p>

If it's the first use, you should be prompted to download the model files. You
can also do it manually, by running `terran checkpoint list` and then
`terran checkpoint download <checkpoint-id>` in a terminal.

Or maybe:

```python
>>> from terran.vis import vis_poses
>>> from terran.pose import pose_estimation
>>>
>>> image = open_image('examples/readme/many-poses-raw.jpg')
>>> display_image(vis_poses(image, pose_estimation(image)))
```

<p align="center">
  <img src="examples/readme/many-poses.jpg", width="720">
</p>

# Examples

## Finding a person in a group of images

`reference` the path to the reference person, which should contain only one
person, `image-dir` the path to the directory containing the images to search
in.

```python
from scipy.spatial.distance import cosine

from terran.face import face_detection, extract_features
from terran.io import open_image, resolve_images

reference = open_image(reference)
faces_in_reference = face_detection(reference)
if len(faces_in_reference) != 1:
    click.echo('Reference image must have exactly one face.')
    return
ref_feature = extract_features(reference, faces_in_reference[0])

paths = resolve_images(
    Path(image_dir).expanduser(),
    batch_size=batch_size
)
for batch_paths in paths:
    batch_images = list(map(open_image, batch_paths))
    faces_per_image = face_detection(batch_images)
    features_per_image = extract_features(batch_images, faces_per_image)

    for path, image, faces, features in zip(
        batch_paths, batch_images, faces_per_image, features_per_image
    ):
        for face, feature in zip(faces, features):
            if cosine(ref_feature, feature) < 0.5:
                display_image(vis_faces(image, face))
```

## Face detection over a video

```python
from terran.face import face_detection
from terran.io import open_video, write_video
from terran.vis import vis_faces

video = open_video(
    video_path,
    batch_size=batch_size,
    read_for=duration,
    start_time=start_time,
    framerate=framerate,
)

writer = write_video(output_path, copy_format_from=video)

for frames in video:
    faces_per_frame = face_detection(frames)
    for frame, faces in zip(frames, faces_per_frame):
        if not faces:
            continue

        # If you don't call `vis_faces` directly, the rendering will be done
        # in the writing thread, thus not blocking the main program while
        # # drawing.
        writer.write_frame(vis_faces, frame, faces)

writer.close()
```

## Customizing model settings

You might want to customize any of the detection functions (such as
`face_detection`) in order to change e.g. the size images are resized to (in
order to make it run faster). You can do it like so:

```python
from terran.face import Detection

face_detection = Detection(short_side=208)

image = open_image(...)
detections = face_detection(image)
```

# License

Terran is released under the [BSD 3-Clause](LICENSE) license.
