import numpy as np

from PIL import Image


def open_image(uri):
    """Opens and returns image at `uri`.
    Always returns a HxWxC `np.array` containing the image, ready to be
    consumed by any Terran algorithm. If the image is grayscale or has an alpha
    channel, it'll get converted into `RGB`, so the number of channels will
    always be 3.
    Parameters
    ----------
    uri : str
        URI pointing to an image, which may be a filesystem location or a URL.
    Returns
    -------
    numpy.ndarray
        Array of size HxWxC containing the pixel values for the image, as
        numpy.uint8.
    """
    # TODO: Allow `uri` being a list?
    # TODO: Eventually accept a URL.
    image = np.asarray(Image.open(uri).convert('RGB'))

    if len(image.shape) == 2:
        # Grayscale image.
        image = np.stack([image] * 3, axis=-1)

    return image
