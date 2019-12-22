import numpy as np

from pathlib import Path
from PIL import Image
from urllib.parse import urlparse
from urllib.request import urlopen


def open_image(uri):
    """Opens and returns image at `uri`.

    Always returns a HxWxC `np.array` containing the image, ready to be
    consumed by any Terran algorithm. If the image is grayscale or has an alpha
    channel, it'll get converted into `RGB`, so the number of channels will
    always be 3.

    Parameters
    ----------
    uri : str or pathlib.Path
        URI pointing to an image, which may be a filesystem location or a URL.

    Returns
    -------
    numpy.ndarray
        Array of size HxWxC containing the pixel values for the image, as
        numpy.uint8.

    """
    # Check if `uri` is a URL or a filesystem location.
    if isinstance(uri, Path):
        image = Image.open(uri)
    elif urlparse(uri).scheme:
        image = Image.open(urlopen(uri))
    else:
        image = Image.open(Path(uri).expanduser())

    image = np.asarray(image.convert('RGB'))

    if len(image.shape) == 2:
        # Grayscale image, turn it to a rank-3 tensor anyways.
        image = np.stack([image] * 3, axis=-1)

    return image
