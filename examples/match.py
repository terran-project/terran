import click

from pathlib import Path
from scipy.spatial.distance import cosine

from terran.face import extract_features, face_detection
from terran.io import open_image, resolve_images
from terran.vis import display_image, vis_faces


@click.command(name='match-dir')
@click.argument('reference')
@click.argument('image-dir')
@click.option('--batch-size', type=int, default=1)
@click.option('--threshold', type=float, default=0.5)
@click.option('--display', is_flag=True, default=False)
def match_directory(reference, image_dir, batch_size, threshold, display):
    reference = open_image(reference)
    faces_in_reference = face_detection(reference)
    if len(faces_in_reference) != 1:
        click.echo('Reference image must have exactly one face.')
        return
    ref_feature = extract_features(reference, faces_in_reference[0])

    paths = resolve_images(
        Path(image_dir).expanduser(),
        batch_size=batch_size
    )
    for batch_paths in paths:
        batch_images = list(map(open_image, batch_paths))
        faces_per_image = face_detection(batch_images)
        features_per_image = extract_features(batch_images, faces_per_image)

        for path, image, faces, features in zip(
            batch_paths, batch_images, faces_per_image, features_per_image
        ):
            for face, feature in zip(faces, features):
                confidence = cosine(ref_feature, feature)
                if confidence < threshold:
                    click.echo(f'{path}, confidence = {confidence:.2f}')
                    if display:
                        display_image(vis_faces(image, face))


if __name__ == '__main__':
    match_directory()
