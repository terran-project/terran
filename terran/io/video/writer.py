import ffmpeg
import os
import subprocess

from threading import Thread
from queue import Queue

from terran.io.video import DEFAULT_WRITER_BUFFER_SIZE, VideoClosed
from terran.io.video.reader import Video, open_video


def _frame_writer(queue, cmd):
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    while True:
        # TODO: Check for changing sizes in the _frame_writer func.
        frame_or_func = queue.get()
        if frame_or_func is None:
            break

        frame_or_func, *args = frame_or_func
        if hasattr(frame_or_func, '__call__'):
            frame = frame_or_func(*args)
        else:
            frame = frame_or_func

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()

    # If process hasn't ended yet, terminate it, giving it some time to finish
    # up the file.
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10.0)


class VideoWriter:

    def __init__(
        self, output_path, framerate=None, copy_format_from=None,
        size_hint=None, **kwargs
    ):
        """Initialize the writing of a video.

        Video format (codecs, framerate, etc.) can be specified using the
        keyword arguments, which will be fed to `ffmpeg` as-is, or by copying
        them from an existing `Video`.

        Parameters
        ----------
        framerate (int, str or None): Framerate of the output video. Will pass
            along to `ffmpeg`.  Can be an `int` with the frames per second, a
            `str` with a fraction (e.g. `'5000/1001'`). Default is `30`, or the
            framerate of `copy_format_from`, if specified.
        copy_format_from : str, pathlib.Path or Video
            Either `Video` instance or path to video to copy format from.
        size_hint : tuple
            (height, width) tuple indicating the frame size the video will
            have. Defaults to `None`, meaning that it will be infered from the
            first frame.

        """
        self.output_path = os.path.expanduser(output_path)

        # Calculate the framerate for the video. Priority is: value of
        # `framerate`, value from `copy_format_from`, or default of `30`.
        if framerate is None and copy_format_from is None:
            self.framerate = 30
        elif framerate is None and copy_format_from is not None:
            if not isinstance(copy_format_from, Video):
                # We got a path to a video, open it to check the framerate.
                copy_format_from = open_video(copy_format_from)
            self.framerate = copy_format_from.framerate
        else:
            self.framerate = framerate

        self.size_hint = size_hint

        # Handler for video-writing thread and queue.
        self._thread = None
        self._queue = None
        self._closed = False

    def __del__(self):
        if not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _prepare_ffmpeg_cmd(self):
        spec = ffmpeg

        kwargs = {
            'framerate': str(self.framerate),
            'format': 'rawvideo',
            'pix_fmt': 'rgb24',
            's': '{}x{}'.format(self.width, self.height),  # Frame size.
        }
        spec = spec.input('pipe:', **kwargs)

        kwargs = {
            'pix_fmt': 'yuv420p',
        }
        spec = spec.output(self.output_path, **kwargs)

        spec = spec.global_args('-y')

        spec = spec.compile()

        return spec

    def write_frame(self, frame_or_func, *args):
        """Write a frame or the result of a rendering function to the video.

        Note that if no `size_hint` is provided when creating the writer, and a
        rendering function is passed as argument, the rendering function might
        be executed twice, in order to infer the size of the video to write.

        """
        if self._closed:
            raise VideoClosed('The video has already been closed.')

        if not self._thread:
            # If no size hint is provided, get the dimensions of the video from
            # the first frame fed. If it's a function, we need to execute it
            # once.
            if not self.size_hint:
                if hasattr(frame_or_func, '__call__'):
                    frame = frame_or_func(*args)
                else:
                    frame = frame_or_func
                self.height, self.width = frame.shape[0:2]
            else:
                self.height, self.width = self.size_hint

            cmd = self._prepare_ffmpeg_cmd()

            self._queue = Queue(DEFAULT_WRITER_BUFFER_SIZE)
            # TODO: Daemon or not?
            self._thread = Thread(
                target=_frame_writer,
                args=(self._queue, cmd)
            )
            self._thread.start()

        self._queue.put((frame_or_func, *args))

    def close(self):
        if self._closed:
            raise VideoClosed('The video has already been closed.')

        if self._thread:
            self._queue.put(None)
            self._thread.join()
            self._closed = True


def write_video(*args, **kwargs):
    """Creates a writer for a video.

    Arguments are passed verbatim to `VideoWriter`.

    Returns
    -------
    VideoWriter
        `VideoWriter` instance representing the writer object that can be fed
        frames.

    """
    return VideoWriter(*args, **kwargs)
