import click

from terran.face import face_detection
from terran.io import open_video, write_video
from terran.vis import vis_faces


@click.command(name='find-video')
@click.argument('video-path')
@click.argument('output-path')
@click.option('--threshold', type=float, default=0.5)
@click.option('--batch-size', default=32)
@click.option('--duration', '-d', default=None, type=int)
@click.option('--framerate', '-f', default=None, type=int)
@click.option('--start-time', '-ss', default=None, type=str)
def find_video(
    video_path, output_path, threshold, batch_size, duration, framerate,
    start_time
):
    # Open video to search in.
    video = open_video(
        video_path,
        batch_size=batch_size,
        read_for=duration,
        start_time=start_time,
        framerate=framerate,
    )

    # Create the video writer, copying the format options such as framerate
    # from `video`.
    writer = write_video(output_path, copy_format_from=video)

    # Iterate over batches of video frames.
    with click.progressbar(video, length=len(video)) as bar:
        for frames in bar:
            faces_per_frame = face_detection(frames)

            for frame, faces in zip(frames, faces_per_frame):
                # If you don't call `vis_faces` directly, the rendering will be
                # done in the writing thread, thus not blocking the main
                # program while drawing.
                writer.write_frame(vis_faces, frame, faces)

    writer.close()


if __name__ == '__main__':
    find_video()
