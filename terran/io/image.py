import numpy as np
import requests

from io import BytesIO
from pathlib import Path
from PIL import Image
from urllib.parse import urlparse


# Use a Chrome-based user agent to avoid getting needlessly blocked.
USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/51.0.2704.103 Safari/537.36'
)


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
        response = requests.get(uri, headers={'User-Agent': USER_AGENT})
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(Path(uri).expanduser())

    image = np.asarray(image.convert('RGB'))

    if len(image.shape) == 2:
        # Grayscale image, turn it to a rank-3 tensor anyways.
        image = np.stack([image] * 3, axis=-1)

    return image


def resolve_images(path, batch_size=None):
    """Collects the paths of all images under `path`, yielding them in batches.

    Ensures that the image is valid before returning it by attempting to open
    it with PIL.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to recursively search for images in.

    Yields
    ------
    pathlib.Path or [pathlib.Path]
        Path to every valid image found under `path`. If `batch_size` is
        `None`, will return a single `pathlib.Path`. Otherwise, returns a list.

    """
    if not isinstance(path, Path):
        path = Path(path).expanduser()

    batch = []
    for f in path.glob('**/*'):
        if not f.is_file():
            continue

        try:
            Image.open(f).verify()
        except OSError:
            continue

        # If no `batch_size` specified, just return the path.
        if batch_size is None:
            yield path.joinpath(f)
            continue

        batch.append(path.joinpath(f))
        if len(batch) >= batch_size:
            yield batch
            batch = []
