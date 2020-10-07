import ffmpeg
import json
import math
import numpy as np
import os
import subprocess

from threading import Event, Thread
from queue import Full as QueueFull, Queue

from terran.io.video import DEFAULT_READER_BUFFER_SIZE, EndOfVideo, VideoClosed


def youtube_dl_available():
    """Check if `youtube-dl` is available in the installation."""
    try:
        import youtube_dl  # noqa
        return True
    except ImportError:
        return False


def ffmpeg_probe(path, **kwargs):
    """Run ffprobe on the specified file and return a JSON representation of
    the output.

    Based on the `ffmpeg.probe` provided by `ffmpeg-python`, but allows passing
    along additional options to `ffmpeg`.

    Parameters
    ----------
    path : str
        This parameter can be a path or URL pointing directly to a video file
        or stream.

    Raises
    ------
    ffmpeg.Error
        If `ffprobe` returns a non-zero exit code. The stderr output can be
        retrieved by accessing the `stderr` property of the exception.

    """
    if not is_path_stream(path):
        path = os.path.expanduser(path)

    # Gather all `kwargs` into a list of tokens to send to `ffprobe`.
    additional_args = []
    for key, value in kwargs.items():
        if not key.startswith('-'):
            key = f'-{key}'
        additional_args.extend([key, str(value)])

    args = [
        'ffprobe', *additional_args, '-show_format', '-show_streams', '-of',
        'json', path
    ]

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    out, err = proc.communicate()
    if proc.returncode != 0:
        raise ffmpeg.Error('ffprobe', out, err)

    return json.loads(out.decode('utf-8'))


def is_path_stream(path):
    """Returns True if the path points to a video stream."""
    return any([
        path.startswith(prefix)
        for prefix in ['/dev/', 'http://', 'https://']
    ])


def parse_timestamp(timestamp):
    if '.' in timestamp:
        timestamp, miliseconds = timestamp.split('.')
        miliseconds = float('0.{}'.format(miliseconds))
    else:
        miliseconds = 0.0
    hours, minutes, seconds = map(float, timestamp.split(':'))
    time = hours * 60 * 60 + minutes * 60 + seconds + miliseconds
    return time


def _read_batch_from_stream(proc, spec):
    width = spec['width']
    height = spec['height']
    batch_size = spec['batch_size']

    # `rgb24` uses three bytes per pixel.
    bytes_to_read = width * height * 3 * (
        batch_size if batch_size is not None else 1
    )
    buffer = proc.stdout.read(bytes_to_read)
    if len(buffer) == 0:
        # No data read, we reached the end of the video.
        return
    elif len(buffer) < bytes_to_read:
        # Received less bytes than expected, must've been unable to
        # complete a batch. Calculate the number of frames read.
        frames_read = int(len(buffer) / width / height / 3)
    else:
        # Received the expected amount data, continue onwards.
        frames_read = batch_size

    frames = np.frombuffer(buffer, np.uint8)
    if batch_size is not None:
        frames = frames.reshape(
            [frames_read, height, width, 3]
        )
    else:
        frames = frames.reshape([height, width, 3])

    return frames


def _clean_up_proc(proc):
    if proc.poll() is None:
        # It's a reader process, we don't care too much for it to end cleanly.
        proc.kill()


def _frame_reader(queue, should_stop, cmd, spec):
    """Worker function for the reading thread."""
    # Open the `ffmpeg` subprocess.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    while proc.poll() is None:
        frames = _read_batch_from_stream(proc, spec)
        if frames is None:
            break

        # Since there's no way to wait on either a `queue.put()` or a stop
        # signal, we must work around it by using a timeout on the
        # `queue.put()` function and periodically checking for the stop signal.
        while True:
            # Stop signal set, clean up the underlying process and return.
            if should_stop.is_set():
                _clean_up_proc(proc)
                return

            # Attempt to place batch in frame, blocking up to one second. If
            # queue is full, try again; if not, break the loop and continue
            # with the next batch.
            try:
                queue.put(frames, timeout=1.0)
            except QueueFull:
                continue

            break

    # After all frames are read, add a sentinel to the queue and terminate the
    # process (if it hasn't already).
    queue.put(None)
    _clean_up_proc(proc)


class Video:
    """Container class for a video, for interfacing with an underlying input.

    Is tasked with handling the file descriptiors, with buffering the stream,
    with dropping frames when needed, with decoding frames out of the video,
    and more.
    """

    def __init__(
        self, path, batch_size=None, framerate=None, is_stream=None,
        read_for=None, start_time=None, ydl_format='best'
    ):
        """Initializes the reading of the video, performing the necessary
        checks.

        Note that if `duration` is `None` and `path` points to a stream,
        `Video` will return frames indefinitely.

        Parameters
        ----------
        path : str
            Path to a video file, stream URL or capture device.
        batch_size : int or None
            Batch size for the returned frames. If `None`, no batching will
            occur (thus, a rank 3 array will be returned).
        framerate : int, str or None
            Framerate to output frames as. Will pass along to `ffmpeg`, which
            will drop or add frames as needed.  Can be an `int` with the frames
            per second, a `str` with a fraction (e.g. `'5000/1001'`) or `None`
            to leave as-is.
        is_stream : bool or None
            Whether the video is a capture device, a stream or somehow a
            never-ending video file. If `None`, will try to guess.
        read_for : int, float or None
            Maximum number of seconds to read, or `None` for reading until the
            end.

            If `read_for` is bigger than the source duration (considering
            the start time), the actual video duration might be less than
            this value.
        start_time : int, str or None
            Time to start video from. If an `int`, specified in seconds. If
            `str`, specified as a timestamp with the format `HH:MM:SS.ms`.
        ydl_format : str
            The format filtering option for YouTube-DL. Check out the
            YouTube-DL "Format Selection" documentation for more information:
            https://github.com/ytdl-org/youtube-dl#format-selection

        """
        self.path = os.path.expanduser(path)

        self.batch_size = batch_size
        self.read_for = read_for
        self._framerate = framerate
        self.ydl_format = ydl_format

        # Parse before storing, so `start_time` will be either an `int`
        # specifying the seconds or `None`.
        if isinstance(start_time, str):
            start_time = parse_timestamp(start_time)
        self.start_time = start_time

        # Try to guess if it's capture device or a stream, as we might need to
        # do things differently.
        if not is_stream:
            self.is_stream = is_path_stream(self.path)

        # Check for existing video streams within the file first, to validate
        # everything and to get the video dimensions.
        try:
            if self.is_stream:
                self.stream_path = self._get_stream_path()

                # If a capture device, use a bigger `probesize` and
                # `analyzeduration` to make sure we have information on all
                # existing streams.
                # TODO: Use a lower probesize and analyzeduration and keep
                # increasing on retries.
                probe = ffmpeg_probe(
                    self.stream_path,
                    probesize=20 * 1024 * 1024,
                    analyzeduration=10 * 1000 * 1000,
                )
            else:
                probe = ffmpeg_probe(self.path)
        except ffmpeg.Error:
            message = f'Video at `{path}` not found. Are you sure it exists?'
            if not youtube_dl_available():
                message += (
                    "\n\n"
                    "Unable to find suitable way to stream from online video "
                    "platforms. If you're trying to stream from YouTube or "
                    "other streaming platforms, make sure `youtube-dl` is "
                    "installed first. If not, ignore this message."
                )

            raise ValueError(message)

        video_stream = next(
            (
                stream for stream in probe['streams']
                if stream.get('codec_type') == 'video'
            ), None
        )
        if not video_stream:
            raise ValueError(
                f'No video stream found at `{path}`. Are you sure this is a '
                'video file or stream?'
            )

        # Extract the information we need from the underlying source video.
        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])

        # Get the framerate of the video source. Uses `avg_frame_rate`, as
        # `r_frame_rate` will default to the lowest common denominator when
        # multiple streams are present (such as on DVB devices). Should be
        # accurate enough.
        if '/' in video_stream['avg_frame_rate']:
            num, den = map(int, video_stream['avg_frame_rate'].split('/'))
            self.source_framerate = num / den
        else:
            self.source_framerate = float(video_stream['avg_frame_rate'])

        # Attempt to get the duration either from the selected video stream or
        # from the container data.
        self.source_duration = None
        if 'duration' in video_stream:
            self.source_duration = float(video_stream['duration'])
        elif 'duration' in probe.get('format', {}):
            self.source_duration = float(probe['format']['duration'])

        if self.duration is not None and self.duration < 0:
            raise ValueError(
                'Duration of the video is negative. Is the `start_time` '
                'timestamp after the video ends?'
            )

        # Handler for video-reading thread and queue.
        self._thread = None
        self._queue = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            frames = self.read_frames()
        except EndOfVideo:
            raise StopIteration
        return frames

    def __del__(self):
        if not self._closed:
            self.close()

    def __len__(self):
        """Returns number of batches present in the video.

        Raises
        ------
        AttributeError
            If the video is a stream or we were otherwise unable to infer the
            video length.

        """
        if not self.duration:
            raise AttributeError(
                'Video doesn\'t have a duration. Is it a stream?'
            )

        batch_size = self.batch_size if self.batch_size else 1
        return math.ceil(
            math.ceil(self.duration * self.framerate) / batch_size
        )

    @property
    def framerate(self):
        """Effective framerate of the video output.

        Either the requested framerate, if set, or the source's.
        """
        return (
            self._framerate if self._framerate is not None
            else self.source_framerate
        )

    @property
    def duration(self):
        """Effective duration of the video output, in seconds.

        Considers the `read_for` attribute, the duration of the source video,
        and the `start_time` set by the user.

        Returns
        -------
        float or None
            Time in seconds or `None` if it's a stream.

        """
        if not self.source_duration:
            # If no duration, will read for `self.read_for` seconds, which may
            # be `None`.
            return self.read_for

        # If source duration is available, consider the time that was skipped
        # due to `self.start_time`.
        source_duration = (
            self.source_duration if not self.start_time
            else self.source_duration - self.start_time
        )
        if self.read_for:
            return min(source_duration, self.read_for)
        else:
            return source_duration

    def _get_stream_path(self):
        """Check if video stream comes from a video sharing platform"""
        if not youtube_dl_available():
            return self.path

        import youtube_dl

        ydl_options = {
            'format': self.ydl_format,
            'quiet': True,
            'no_warnings': True
        }

        for extractor in youtube_dl.gen_extractors():
            if extractor.suitable(self.path):
                try:
                    with youtube_dl.YoutubeDL(ydl_options) as ydl:
                        stream_info = ydl.extract_info(
                            self.path, download=False
                        )
                        self.ydl_info = stream_info

                        if stream_info['url'] is None:
                            raise ValueError(
                                'Unable to find stream URL for video format '
                                f'{self.ydl_format}'
                            )
                        return stream_info['url']
                except youtube_dl.utils.YoutubeDLError:
                    break

        return self.path

    def _prepare_ffmpeg_cmd(self):
        """Prepare the subprocess command for ffmpeg."""
        spec = ffmpeg

        # Prepare the input parameters.
        kwargs = {
            'err_detect': 'ignore_err',
        }

        if self.duration is not None:
            kwargs.update({'t': str(self.duration)})

        if self.is_stream:
            # If a capture device, use a bigger `probesize` and
            # `analyzeduration` to make sure we have information on all
            # existing streams.
            kwargs.update({
                'probesize': 20 * 1024 * 1024,
                'analyzeduration': 10 * 1000 * 1000,
            })

        # Start time works for normal videos and YouTube streams
        if self.start_time or self.is_stream:
            # Seek after 5 seconds to allow ffmpeg to see a reference frame.
            # TODO: Why is this even necessary? Can't I just tell it to start
            # from where valid?
            kwargs.update({'ss': self.start_time or '00:00:05'})

        # Send the appropriate path to `ffmpeg`, either stream or normal.
        input_file = self.stream_path if self.is_stream else self.path
        spec = spec.input(input_file, **kwargs)

        # Prepare the output parameters.
        kwargs = {}

        # Only pass the flag if a specific framerate was requested by the user.
        if self._framerate:
            kwargs.update({'r': str(self._framerate)})

        spec = spec.output(
            'pipe:', format='rawvideo', pix_fmt='rgb24', **kwargs
        )

        cmd = spec.compile()
        return cmd

    def read_frames(self):
        # If the video resource has been opened and closed already, bail out.
        if self._closed:
            raise EndOfVideo

        # Open and manage the `ffmpeg` process explicitly for better
        # control. The process will be opened in a `Thread` (as it's mostly
        # I/O-bound and a thread provides a lower overhead) and will prefetch
        # `DEFAULT_READER_BUFFER_SIZE` batches in advance.
        if not self._thread:
            cmd = self._prepare_ffmpeg_cmd()
            spec = {
                'width': self.width,
                'height': self.height,
                'batch_size': self.batch_size,
            }

            self._queue = Queue(DEFAULT_READER_BUFFER_SIZE)
            self._stop_signal = Event()

            self._thread = Thread(
                args=(self._queue, self._stop_signal, cmd, spec),
                name='FrameReader',
                target=_frame_reader,
                # We don't want to keep the program hanging; as it's a reader
                # thread, we don't care that the resources are closed abruptly.
                daemon=True,
            )
            self._thread.start()

        frames = self._queue.get()
        if frames is None:
            raise EndOfVideo

        return frames

    def close(self):
        if self._closed:
            raise VideoClosed('The video has already been closed.')

        if self._thread:
            # Set the stop signal, making the thread stop trying to put frames
            # into the queue. Due to the hackish approach, it might take up to
            # one second to finish.
            self._stop_signal.set()
            self._thread.join()
            self._closed = True


def open_video(*args, **kwargs):
    """Opens a video file or stream.

    The function will guess whether the input is a video file, a capture device
    or a stream and will proceed as needed.

    Arguments are passed verbatim to `Video`.

    Returns
    -------
    Video
        `Video` instance representing the video at `path`.

    """
    return Video(*args, **kwargs)
