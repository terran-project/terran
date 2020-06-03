import cairo
import math
import numpy as np

from cairo import Context, ImageSurface
from functools import wraps

from terran.pose import Keypoint
from terran.vis import (
    FACE_COLORMAP, MARKER_SCALES, POSE_CONNECTIONS, POSE_CONNECTION_COLORS,
    POSE_KEYPOINT_COLORS,
)


def with_cairo(vis_func):
    """Wrapper function to prepare the cairo context for the vis function."""

    @wraps(vis_func)
    def func(image, objects, *args, **kwargs):
        # Allow sending in a single object in every function.
        if not (isinstance(objects, list) or isinstance(objects, tuple)):
            objects = [objects]

        # Calculate the appropriate scaling for the markers.
        area = image.shape[1] * image.shape[0]
        for ref_area, scale in MARKER_SCALES:
            if area >= ref_area:
                break

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

        # Set up the font. If not available, will default to a Sans Serif
        # font available in the system, so no need to have fallbacks.
        ctx.select_font_face(
            "DejaVuSans-Bold",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        ctx.set_font_size(int(16 * scale))

        vis_func(ctx, objects, scale=scale, *args, **kwargs)

        # Return the newly-drawn image, excluding the extra alpha channel
        # added.
        image = with_alpha[..., :-1][..., ::-1]

        return image

    return func


def draw_marker(ctx, coords, color=(255, 0, 0), scale=1):
    """Draw a marker on `ctx` at `coords`.

    The marker itself is a rectangle with rounded corners.
    """
    x_min, y_min, x_max, y_max = coords
    width = x_max - x_min
    height = y_max - y_min

    degrees = math.pi / 180.0

    radius = 10.0 * scale
    ctx.set_source_rgba(*color, 1.0)
    ctx.set_line_width(3. * scale)
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

    ctx.set_dash([10. * scale])
    ctx.set_line_width(1. * scale)

    ctx.move_to((x_min + x_max) / 2, y_min)
    ctx.line_to((x_min + x_max) / 2, y_max)

    ctx.move_to(x_min, (y_min + y_max) / 2)
    ctx.line_to(x_max, (y_min + y_max) / 2)

    ctx.stroke()


@with_cairo
def vis_faces(ctx, faces, scale=1.0):
    """Draw boxes over the detected faces for the given image.

    Parameters
    ----------
    image : np.ndarray representing an image
        Image to draw faces over.
    faces : dict or list of dicts (from `face_detection` or `face_tracking`)
        Faces to draw on `image`. The expected format is the one returned from
        `face_detection` or `face_tracking`, with two optional extra fields:

        - ``text`` (str): Text to be written next to the box.
        - ``name`` (str): Name associated to the face, in order to make the
          color used for the box fixed.

        If available, the ``track`` field will be used as the default value
        for the two values above, if they aren't specified.

    Returns
    -------
    np.ndarray
        Copy of `image` with the faces drawn over.

    """
    for face in faces:
        # Get the name and text for the current face.
        face_name = face.get('name') or face.get('track')
        if face.get('text') is not None:
            face_text = face['text']
        elif face.get('track') is not None:
            face_text = f"#{face['track']}"
        else:
            face_text = None

        color = map(lambda x: x / 255, FACE_COLORMAP(face_name))
        draw_marker(ctx, face['bbox'], color=color, scale=scale)

        if face_text is not None:
            ctx.move_to(
                face['bbox'][0] + 3 * scale,
                face['bbox'][1] + 15 * scale
            )
            ctx.show_text(face_text)


def draw_keypoints(ctx, keypoints, scale=1.0):
    for keypoint in keypoints:
        for idx, (x, y, is_present) in enumerate(keypoint['keypoints']):
            if not is_present:
                continue

            color = map(
                lambda x: x / 255,
                POSE_KEYPOINT_COLORS[Keypoint(idx)]
            )
            ctx.set_source_rgba(*color, 0.9)

            ctx.arc(x, y, 3 * scale, 0, 2 * math.pi)
            ctx.fill()
            ctx.stroke()


def draw_limbs(ctx, keypoints, scale=1.0):
    for keypoint in keypoints:
        kps = keypoint['keypoints']
        for idx, (conn_src, conn_dst) in enumerate(POSE_CONNECTIONS):
            x_src, y_src, src_present = kps[conn_src.value]
            x_dst, y_dst, dst_present = kps[conn_dst.value]

            # Ignore limbs for which one of the keypoints is missing.
            if not (src_present and dst_present):
                continue

            color = map(lambda x: x / 255, POSE_CONNECTION_COLORS[idx])
            ctx.set_source_rgba(*color, 0.7)
            ctx.set_line_width(1.)

            # We'll use a Bezier curve using a rectangle around the line as
            # control points, so we first calculate the normal and the
            # direction we need to move the points on in order to have a
            # constant-height box around the lines.
            width = 4 * scale

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
def vis_poses(ctx, poses, scale=1.0):
    """Draw boxes over the detected poses for the given image.

    Parameters
    ----------
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
    draw_limbs(ctx, poses, scale=scale)
    draw_keypoints(ctx, poses, scale=scale)
