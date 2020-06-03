import numpy as np
import random

from PIL import Image
from io import BytesIO
from subprocess import run, SubprocessError

from terran.pose import Keypoint


MARKER_SCALES = [
    (1920 * 1080, 1.8),
    (1280 * 720, 1.5),
    (480 * 360, 1.3),
    (0, 1),
]


def display_image(image):
    """Displays an image using an external viewer.

    Will first try using `feh`, if it's installed locally, and `matplotlib` as
    fallback, if an error occurs.

    Parameters
    ----------
    image : np.array or PIL.Image
        Image to be displayed.

    Raises
    ------
    Exception
        If no suitable backend is found.

    """
    # If a numpy array, turn into a Pillow image first.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Save the image compressed in a `BytesIO`.
    buf = BytesIO()
    image.save(buf, format='png')
    buf.seek(0)

    # Run the `feh` command, passing the buffer as input. If any error occurs,
    # fallback to using `imshow` from `matplotlib`.
    try:
        run(['feh', '-'], input=buf.read())
    except (FileNotFoundError, SubprocessError):
        try:
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.show()
        except ImportError:
            raise Exception(
                'Unable to find a suitable backend to display an image. '
                'Tried `feh` and `matplotlib`. Install either in order to use '
                'this function.'
            )


def hex_to_rgb(x):
    """Turns a color hex representation into a tuple representation."""
    return tuple([int(x[i:i + 2], 16) for i in (0, 2, 4)])


def build_colormap():
    """Builds a colormap function that maps labels to colors.

    Returns:
        Function that receives a label and returns a color tuple `(R, G, B)`
        for said label.
    """
    # Build the 10-color palette to be used for all classes. The following are
    # the hex-codes for said colors (taken the default 10-categorical d3 color
    # palette).
    palette = (
        '1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf'
    )
    colors = [hex_to_rgb(palette[i:i + 6]) for i in range(0, len(palette), 6)]

    seen_labels = {}

    def colormap(label=None):
        # If `label` is `None` return a random color, so we can have both fix
        # the colors and have them dynamic.
        if label is None:
            return random.choice(colors)

        # If label not yet seen, get the next value in the palette sequence.
        if label not in seen_labels:
            seen_labels[label] = colors[len(seen_labels) % len(colors)]

        return seen_labels[label]

    return colormap


FACE_COLORMAP = build_colormap()


POSE_CONNECTIONS = [
    (Keypoint.NOSE, Keypoint.NECK),
    (Keypoint.NOSE, Keypoint.R_EYE), (Keypoint.R_EYE, Keypoint.R_EAR),
    (Keypoint.NOSE, Keypoint.L_EYE), (Keypoint.L_EYE, Keypoint.L_EAR),

    (Keypoint.NECK, Keypoint.R_SHOULDER),
    (Keypoint.R_SHOULDER, Keypoint.R_ELBOW),
    (Keypoint.R_ELBOW, Keypoint.R_HAND),

    (Keypoint.NECK, Keypoint.R_HIP),
    (Keypoint.R_HIP, Keypoint.R_KNEE),
    (Keypoint.R_KNEE, Keypoint.R_FOOT),

    (Keypoint.NECK, Keypoint.L_SHOULDER),
    (Keypoint.L_SHOULDER, Keypoint.L_ELBOW),
    (Keypoint.L_ELBOW, Keypoint.L_HAND),

    (Keypoint.NECK, Keypoint.L_HIP),
    (Keypoint.L_HIP, Keypoint.L_KNEE),
    (Keypoint.L_KNEE, Keypoint.L_FOOT),
]


POSE_CONNECTION_COLORS = list(map(hex_to_rgb, [
    # Head.
    'e6550d', 'fd8d3c', 'fdae6b', '843c39', 'ad494a',

    # Right side.
    '637939', '8ca252', 'b5cf6b',
    '843c39', 'ad494a', 'd6616b',

    # Left side.
    '3182bd', '6baed6', '9ecae1',
    '8c6d31', 'bd9e39', 'e7ba52',
]))


POSE_KEYPOINT_COLORS = {
    Keypoint.NOSE: hex_to_rgb('e6550d'),
    Keypoint.NECK: hex_to_rgb('fd8d3c'),
    Keypoint.R_EYE: hex_to_rgb('fdae6b'),
    Keypoint.L_EYE: hex_to_rgb('843c39'),
    Keypoint.R_EAR: hex_to_rgb('ad494a'),
    Keypoint.L_EAR: hex_to_rgb('d6616b'),

    Keypoint.R_SHOULDER: hex_to_rgb('637939'),
    Keypoint.R_ELBOW: hex_to_rgb('8ca252'),
    Keypoint.R_HAND: hex_to_rgb('b5cf6b'),
    Keypoint.R_HIP: hex_to_rgb('843c39'),
    Keypoint.R_KNEE: hex_to_rgb('ad494a'),
    Keypoint.R_FOOT: hex_to_rgb('d6616b'),

    Keypoint.L_SHOULDER: hex_to_rgb('3182bd'),
    Keypoint.L_ELBOW: hex_to_rgb('6baed6'),
    Keypoint.L_HAND: hex_to_rgb('9ecae1'),
    Keypoint.L_HIP: hex_to_rgb('8c6d31'),
    Keypoint.L_KNEE: hex_to_rgb('bd9e39'),
    Keypoint.L_FOOT: hex_to_rgb('e7ba52'),
}


try:
    from terran.vis.cairo import (  # noqa
        vis_poses, vis_faces,
    )
except ImportError:
    from terran.vis.pillow import (  # noqa
        vis_poses, vis_faces,
      )
