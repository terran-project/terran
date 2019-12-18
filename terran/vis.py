import cairo
import math
import numpy as np

from cairo import Context, ImageSurface
from PIL import Image
from io import BytesIO
from subprocess import run, SubprocessError


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


def vis_faces(image, faces):
    if not (isinstance(faces, list) or isinstance(faces, tuple)):
        faces = [faces]

    # TODO: Ideally, we would like to avoid having to create a copy of the
    # array just to add paddings to support the cairo format, but there's no
    # way to use a cairo surface with 24bpp.
    # TODO: Take into account endianness of the machine, as it's possible it
    # has to be concatenated in the other order in some machines.

    # We need to add an extra `alpha` layer, as cairo only supports 32 bits per
    # pixel formats, and our numpy array uses 24. This means creating one copy
    # before modifying and a second copy afterwards.
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

    # Draw the markers around the faces found.
    for face in faces:
        # Draw a circle within the bounding box.
        color = map(lambda x: x / 255, colormap(face.get('name', '')))
        draw_marker(ctx, face['bbox'], color=color)

        if face.get('text'):
            ctx.move_to(face['bbox'][0] + 3, face['bbox'][1] + 20)
            ctx.show_text(face['text'])

    # Return the newly-drawn image, excluding the extra alpha channel added.
    image = with_alpha[..., :-1][..., ::-1]

    return image
