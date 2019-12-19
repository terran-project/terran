import cairo
import math
import numpy as np

from cairo import Context, ImageSurface
from PIL import Image
from io import BytesIO
from subprocess import run, SubprocessError
from functools import wraps


def with_cairo(vis_func):
    """Wrapper function to prepare the cairo context for the vis function."""

    @wraps(vis_func)
    def func(image, objects, *args, **kwargs):
        # Allow sending in a single object in every function.
        if not (isinstance(objects, list) or isinstance(objects, tuple)):
            objects = [objects]

        # TODO: Ideally, we would like to avoid having to create a copy of the
        # array just to add paddings to support the cairo format, but there's
        # no way to use a cairo surface with 24bpp.

        # TODO: Take into account endianness of the machine, as it's possible
        # it has to be concatenated in the other order in some machines.

        # We need to add an extra `alpha` layer, as cairo only supports 32 bits
        # per pixel formats, and our numpy array uses 24. This means creating
        # one copy before modifying and a second copy afterwards.
        with_alpha = np.concatenate(
            [
                image[..., ::-1],
                255 * np.ones(
                    (image.shape[0], image.shape[1], 1),
                    dtype=np.uint8
                )
            ], axis=2
        )

        surface = ImageSurface.create_for_data(
            with_alpha,
            cairo.Format.RGB24,
            image.shape[1],
            image.shape[0]
        )

        ctx = Context(surface)

        # Set up the font.
        # TODO: What if it doesn't exist? Can I have fallbacks?
        ctx.select_font_face(
            "DejaVuSans-Bold",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        ctx.set_font_size(16)

        vis_func(ctx, objects, *args, **kwargs)

        # Return the newly-drawn image, excluding the extra alpha channel
        # added.
        image = with_alpha[..., :-1][..., ::-1]

        return image

    return func


def display_image(image):
    """Displays an image using `feh`.

    Arguments:
        image (np.array or PIL.Image): Image to be displayed.
    """
    # TODO: Can I pass several images to `feh` (e.g. the whole batch at once)?

    # If a numpy array, turn into a Pillow image first.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Save the image compressed in a `BytesIO`.
    buf = BytesIO()
    image.save(buf, format='png')
    buf.seek(0)

    # Run the `feh` command, passing the buffer as input. If any error occurs,
    # just ignore it, as it may simply not be supported in platform.
    try:
        run(['feh', '-'], input=buf.read())
    except SubprocessError as e:
        print('Error displaying image:', e)


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

    def colormap(label):
        # If label not yet seen, get the next value in the palette sequence.
        if label not in seen_labels:
            seen_labels[label] = colors[len(seen_labels) % len(colors)]

        return seen_labels[label]

    return colormap


# TODO: Remove from here.
colormap = build_colormap()


def draw_marker(ctx, coords, color=(255, 0, 0), radius=10.0):
    """Draw a marker on `ctx` at `coords`.

    The marker itself is a rectangle with rounded corners.
    """
    x_min, y_min, x_max, y_max = coords
    width = x_max - x_min
    height = y_max - y_min

    degrees = math.pi / 180.0

    ctx.set_source_rgba(*color, 1.0)
    ctx.set_line_width(3.)
    ctx.set_dash([])

    ctx.new_sub_path()
    ctx.arc(
        x_min + width - radius, y_min + radius,
        radius, -90 * degrees, 0 * degrees
    )
    ctx.arc(
        x_min + width - radius, y_min + height - radius,
        radius, 0 * degrees, 90 * degrees
    )
    ctx.arc(
        x_min + radius, y_min + height - radius,
        radius, 90 * degrees, 180 * degrees
    )
    ctx.arc(
        x_min + radius, y_min + radius,
        radius, 180 * degrees, 270 * degrees
    )
    ctx.close_path()

    ctx.stroke()

    ctx.set_dash([10.])
    ctx.set_line_width(1.)

    ctx.move_to((x_min + x_max) / 2, y_min)
    ctx.line_to((x_min + x_max) / 2, y_max)

    ctx.move_to(x_min, (y_min + y_max) / 2)
    ctx.line_to(x_max, (y_min + y_max) / 2)

    ctx.stroke()


@with_cairo
def vis_faces(ctx, faces):
    """Draw boxes over the detected faces for the given image.

    Paramters
    ---------
    image : np.ndarray representing an image.
        Image to draw faces over.
    faces : dict or list of dicts, as returned by `face_detection`
        Faces to draw on `image`. The expected format is the one returned from
        `face_detection`, with an optional extra field `text`, which will be
        written next to the box, and `name`, to identify the face and make the
        color used fixed.

    Returns
    -------
    np.ndarray
        Copy of `image` with the faces drawn over.

    """
    # Draw the markers around the faces found.
    for face in faces:
        # Draw a circle within the bounding box.
        color = map(lambda x: x / 255, colormap(face.get('name', '')))
        draw_marker(ctx, face['bbox'], color=color)

        if face.get('text'):
            ctx.move_to(face['bbox'][0] + 3, face['bbox'][1] + 20)
            ctx.show_text(face['text'])


def draw_keypoints(ctx, keypoints):

    # TODO: Don't do like this.
    colormap = list(map(
        hex_to_rgb,
        [
            'e6550d', 'fd8d3c',
            '637939', '8ca252', 'b5cf6b',
            '3182bd', '6baed6', '9ecae1',
            '843c39', 'ad494a', 'd6616b',
            '8c6d31', 'bd9e39', 'e7ba52',
            'fdae6b', '843c39', 'ad494a', 'd6616b',
        ]
    ))

    for keypoint in keypoints:
        for idx, (x, y, is_present) in enumerate(keypoint['keypoints']):
            if not is_present:
                continue

            color = map(lambda x: x / 255, colormap[idx])
            ctx.set_source_rgba(*color, 0.9)

            ctx.arc(x, y, 5, 0, 2 * math.pi)
            ctx.fill()
            ctx.stroke()


def draw_limbs(ctx, keypoints):
    # TODO: Don't do like this.
    colormap = list(map(
        hex_to_rgb,
        # Oranges.
        # 'e6550d', 'fd8d3c', 'fdae6b',
        # Blues.
        # '3182bd', '6baed6', '9ecae1',
        # Reds.
        # '843c39', 'ad494a', 'd6616b',
        # Greens.
        # '637939', '8ca252', 'b5cf6b',
        # Yellows.
        # '8c6d31', 'bd9e39', 'e7ba52',
        # Purples.
        # '7b4173', 'a55194', 'ce6dbd',
        # Violets.
        # '393b79', '5253a3', '6b6ecf', '9c9ede',
        [
            'e6550d', 'fd8d3c', 'fdae6b', '843c39', 'ad494a',

            '637939', '8ca252', 'b5cf6b',
            '843c39', 'ad494a', 'd6616b',

            '3182bd', '6baed6', '9ecae1',
            '8c6d31', 'bd9e39', 'e7ba52',
        ]
    ))

    connections = [
        # Head connections.
        (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),

        # Right arm.
        (1, 2), (2, 3), (3, 4),
        # Right foot.
        (1, 8), (8, 9), (9, 10),

        # Left arm.
        (1, 5), (5, 6), (6, 7),
        # Left foot.
        (1, 11), (11, 12), (12, 13),
    ]

    for keypoint in keypoints:
        kps = keypoint['keypoints']
        for idx, (conn_src, conn_dst) in enumerate(connections):
            x_src, y_src, src_present = kps[conn_src]
            x_dst, y_dst, dst_present = kps[conn_dst]

            # Ignore limbs for which one of the keypoints is missing.
            if not (src_present and dst_present):
                continue

            color = map(lambda x: x / 255, colormap[idx])
            ctx.set_source_rgba(*color, 0.7)
            ctx.set_line_width(4.)

            # We'll use a Bezier curve using a rectangle around the line as
            # control points, so we first calculate the normal and the
            # direction we need to move the points on in order to have a
            # constant-height box around the lines.
            width = 7

            if abs(y_dst - y_src) > 0:
                normal = - (x_dst - x_src) / (y_dst - y_src)
                x_base = width / math.sqrt(normal ** 2 + 1)
                y_base = x_base * normal
            else:
                # Careful calculating normal if limb is horizontal.
                x_base = 0
                y_base = width

            ctx.move_to(x_src, y_src)
            ctx.curve_to(
                int(x_src + x_base), int(y_src + y_base),
                int(x_dst + x_base), int(y_dst + y_base),
                x_dst, y_dst,
            )
            ctx.curve_to(
                int(x_dst - x_base), int(y_dst - y_base),
                int(x_src - x_base), int(y_src - y_base),
                x_src, y_src,
            )
            ctx.fill()

            ctx.stroke()


@with_cairo
def vis_poses(ctx, poses):
    """Draw boxes over the detected poses for the given image.

    Paramters
    ---------
    image : np.ndarray representing an image.
        Image to draw faces over.
    poses : dict or list of dicts, as returned by `pose_estimation`
        Poses to draw on `image`. The expected format is the one returned from
        `pose_estimation`.

    Returns
    -------
    np.ndarray
        Copy of `image` with the poses drawn over.

    """
    draw_limbs(ctx, poses)
    draw_keypoints(ctx, poses)
