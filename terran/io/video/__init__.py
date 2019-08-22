"""Module containig I/O-related functions for video handling."""
# Default size for the reader buffer, in batches. A size of 1, which is
# equivalent to prefetching the following batch, seems to be more than enough,
# and doesn't use up too much extra memory (~190MB for a 1080p batch of size
# 32).
DEFAULT_READER_BUFFER_SIZE = 1

# Default size for the writer buffer, in frames.
DEFAULT_WRITER_BUFFER_SIZE = 64


class EndOfVideo(Exception):
    pass


class VideoClosed(Exception):
    pass


from terran.io.video.reader import open_video  # noqa
from terran.io.video.writer import write_video  # noqa
