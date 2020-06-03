import numpy as np
import sys

from PIL import Image, ImageDraw, ImageFont

from terran.pose import Keypoint
from terran.vis import (
    FACE_COLORMAP, POSE_CONNECTIONS, POSE_CONNECTION_COLORS,
    POSE_KEYPOINT_COLORS,
)


def get_font():
    """Attempts to retrieve a reasonably-looking TTF font from the system.

    We don't make much of an effort, but it's what we can reasonably do without
    incorporating additional dependencies for this task.
    """
    if sys.platform == 'win32':
        font_names = ['Arial']
    elif sys.platform in ['linux', 'linux2']:
        font_names = ['DejaVuSans-Bold', 'DroidSans-Bold']
    elif sys.platform == 'darwin':
        font_names = ['Menlo', 'Helvetica']

    font = None
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name)
            break
        except IOError:
            continue

    return font


SYSTEM_FONT = get_font()


def draw_label(draw, coords, text, color, scale=1):
    """Draw a box with the label and probability."""
    # Attempt to get a native TTF font. If not, use the default bitmap font.
    global SYSTEM_FONT
    if SYSTEM_FONT:
        label_font = SYSTEM_FONT.font_variant(size=round(16 * scale))
    else:
        label_font = ImageFont.load_default()

    text = str(text)  # `text` may not be a string.

    # We want the probability font to be smaller, so we'll write the label in
    # two steps.
    text_w, text_h = label_font.getsize(text)

    # Get margins to manually adjust the spacing. The margin goes between each
    # segment (i.e. margin, label, margin, prob, margin).
    margin_w, margin_h = label_font.getsize('M')
    margin_w *= 0.2
    _, full_line_height = label_font.getsize('Mq')

    # Draw the background first, considering all margins and the full line
    # height.
    background_coords = [
        coords[0],
        coords[1],
        coords[0] + text_w + + 3 * margin_w,
        coords[1] + full_line_height * 1.15,
    ]
    draw.rectangle(background_coords, fill=color + (255,))

    # Then write the two pieces of text.
    draw.text([
        coords[0] + margin_w,
        coords[1],
    ], text, font=label_font)


def draw_marker(draw, coords, color=(255, 0, 0), scale=1):
    """Draw a marker on `draw` at `coords`.

    The marker itself is a rectangle with rounded corners.
    """
    scale = int(3 * scale)
    outline = color + (255,)
    draw.rectangle(list(coords), outline=outline, width=scale)


def vis_faces(image, faces, scale=1.0):
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
    if not (isinstance(faces, list) or isinstance(faces, tuple)):
        faces = [faces]

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, 'RGBA')

    for face in faces:
        # Get the name and text for the current face.
        face_name = face.get('name') or face.get('track')
        if face.get('text') is not None:
            face_text = face['text']
        elif face.get('track') is not None:
            face_text = f"#{face['track']}"
        else:
            face_text = None

        color = tuple(FACE_COLORMAP(face_name))
        draw_marker(draw, face['bbox'], color=color, scale=scale)

        if face_text is not None:
            draw_label(
                draw, face['bbox'][:2], face_text, color, scale=scale
            )

    return np.asarray(image)


def draw_keypoints(draw, keypoints, scale=1.0):
    scale = int(scale * 4)

    for keypoint in keypoints:
        for idx, (x, y, is_present) in enumerate(keypoint['keypoints']):
            if not is_present:
                continue

            color = tuple(POSE_KEYPOINT_COLORS[Keypoint(idx)]) + (225,)
            draw.ellipse([
                x - int(3 * scale / 2), y - int(3 * scale / 2),
                x + int(3 * scale / 2), y + int(3 * scale / 2),
            ], fill=color)


def draw_limbs(draw, keypoints, scale=1.0):
    scale = int(scale * 8)

    for keypoint in keypoints:
        kps = keypoint['keypoints']
        for idx, (conn_src, conn_dst) in enumerate(POSE_CONNECTIONS):
            x_src, y_src, src_present = kps[conn_src.value]
            x_dst, y_dst, dst_present = kps[conn_dst.value]

            # Ignore limbs for which one of the keypoints is missing.
            if not (src_present and dst_present):
                continue

            color = tuple(POSE_CONNECTION_COLORS[idx]) + (180,)
            draw.line([x_src, y_src, x_dst, y_dst], fill=color, width=scale)


def vis_poses(image, poses, scale=1.0):
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
    if not (isinstance(poses, list) or isinstance(poses, tuple)):
        poses = [poses]

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, 'RGBA')

    draw_limbs(draw, poses, scale=scale)
    draw_keypoints(draw, poses, scale=scale)

    return np.asarray(image)
