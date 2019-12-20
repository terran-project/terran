import click
import os
import shutil
import requests
import tarfile
import tempfile

from itertools import groupby
from pathlib import Path


DEFAULT_TERRAN_HOME = Path('~/.terran')
CHECKPOINT_PATH = 'checkpoints'

BASE_URL = 'https://github.com/nagitsu/terran/releases/download/v0.0.1-alpha'

LABELS_BY_TASK = {
    'face-detection': 'Face detection (`terran.face.Detection`)',
    'face-recognition': 'Face recognition (`terran.face.Recognition`)',
    'pose-estimation': 'Pose estimation (`terran.pose.Estimation`)',
}

CHECKPOINTS = [
    {
        'id': 'b5d77fff',
        'name': 'RetinaFace',
        'description': (
            'RetinaFace with mnet backbone.'
        ),

        'task': 'face-detection',
        'alias': 'gpu-realtime',

        'performance': 1.0,
        'evaluation': {
            'value': 0.76,
            'metric': 'mAP',
            'is_reported': False,
        },

        'url': f'{BASE_URL}/retinaface-mnet.pth'
    },
    {
        'id': 'b5d77fff',
        'name': 'RetinaFace2',
        'description': (
            'RetinaFace with mnet backbone.'
        ),

        'task': 'face-detection',
        'alias': 'gpu-accurate',

        'performance': 1.0,
        'evaluation': {
            'value': 0.76,
            'metric': 'mAP',
            'is_reported': False,
        },

        'url': f'{BASE_URL}/retinaface-mnet.pth'
    },
    {
        'id': 'b5d77fff',
        'name': 'RetinaFace3',
        'description': (
            'RetinaFace with mnet backbone.'
        ),

        'task': 'face-detection',
        'alias': 'edge',

        'performance': 1.0,
        'evaluation': {
            'value': 0.76,
            'metric': 'mAP',
            'is_reported': False,
        },

        'url': f'{BASE_URL}/retinaface-mnet.pth'
    },
    {
        'id': 'd206e4b0',
        'name': 'ArcFace',
        'description': (
            'ArcFace with Resnet 100 backbone.'
        ),

        'task': 'face-recognition',
        'alias': 'gpu-realtime',

        'performance': 0.9,
        'evaluation': {
            'value': 0.80,
            'metric': 'accuracy',
            'is_reported': False,
        },

        'url': f'{BASE_URL}/arcface-resnet100.pth'
    },
    {
        'id': 'd206e4b0',
        'name': 'AdaCos',
        'description': (
            'ArcFace with Resnet 100 backbone.'
        ),

        'task': 'face-recognition',
        'alias': 'gpu-accurate',

        'performance': 0.9,
        'evaluation': {
            'value': 0.80,
            'metric': 'accuracy',
            'is_reported': False,
        },

        'url': f'{BASE_URL}/arcface-resnet100.pth'
    },
    {
        'id': '11a769ad',
        'name': 'OpenPose',
        'description': (
            'OpenPose with VGG backend, 2017 version. Has some modifications, '
            'improving computational efficiency by giving up mAP.'
        ),

        'task': 'pose-estimation',
        'alias': 'gpu-realtime',

        'performance': 1.8,
        'evaluation': {
            'value': 0.65,
            'metric': 'mAP',
            'is_reported': True,
        },

        'url': f'{BASE_URL}/openpose-body.pth'
    },
    {
        'id': '11a769ad',
        'name': 'AlphaPose',
        'description': (
            'OpenPose with VGG backend, 2017 version. Has some modifications, '
            'improving computational efficiency by giving up mAP.'
        ),

        'task': 'pose-estimation',
        'alias': 'gpu-accurate',

        'performance': 1.8,
        'evaluation': {
            'value': 0.65,
            'metric': 'mAP',
            'is_reported': True,
        },

        'url': f'{BASE_URL}/openpose-body.pth'
    },
    {
        'id': '11a769ad',
        'name': 'PersonLab',
        'description': (
            'OpenPose with VGG backend, 2017 version. Has some modifications, '
            'improving computational efficiency by giving up mAP.'
        ),

        'task': 'pose-estimation',
        'alias': 'edge',

        'performance': 1.8,
        'evaluation': {
            'value': 0.65,
            'metric': 'mAP',
            'is_reported': True,
        },

        'url': f'{BASE_URL}/openpose-body.pth'
    },
]


def get_terran_home(create_if_missing=True):
    """Returns Terran's homedir.

    Defaults to `DEFAULT_TERRAN_HOME`, which is `~/.terran`, but can be
    overridden with the `TERRAN_HOME` environment variable.

    Returns
    -------
    pathlib.Path
        Path pointing to the base Terran directory.

    """
    path = Path(
        os.environ.get('TERRAN_HOME', DEFAULT_TERRAN_HOME)
    ).expanduser()

    # Create the directory if it doesn't exist.
    if create_if_missing and not path.exists():
        path.mkdir(exist_ok=True)

    return path


def get_checkpoints_directory():
    """Returns checkpoint directory within Terran's homedir.

    If the path doesn't exists, creates it.

    Returns
    -------
    pathlib.Path
        Path pointing to the checkpoints directory.

    """
    path = get_terran_home() / CHECKPOINT_PATH
    path.mkdir(exist_ok=True)
    return path


def get_checkpoint_path(checkpoint_id):
    """Returns checkpoint's directory given its ID."""
    return get_checkpoints_directory() / checkpoint_id


# Index-related functions: access and mutation.

def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    checkpoints = [
        {
            'status': 'DOWNLOADED',
            **checkpoint
        }
        for checkpoint in CHECKPOINTS
    ]
    return {
        'checkpoints': checkpoints
    }
    # path = get_checkpoints_directory() / CHECKPOINT_INDEX
    # if not path.exists():
    #     return {'checkpoints': []}

    # with open(path) as f:
    #     index = json.load(f)

    # return index


# def save_checkpoint_db(checkpoints):
#     """Overwrites the database file in disk with `checkpoints`."""
#     path = get_checkpoints_directory() / CHECKPOINT_INDEX
#     with open(path, 'w') as f:
#         json.dump(checkpoints, f)


def get_checkpoint(db, checkpoint_id):
    """Returns checkpoint entry in `db` indicated by `checkpoint_id`."""
    selected = [c for c in db['checkpoints'] if c['id'] == checkpoint_id]

    if len(selected) < 1:
        return None

    if len(selected) > 1:
        click.echo(
            f"Multiple checkpoints found for '{checkpoint_id}' "
            f"({len(selected)}). Returning first."
        )

    return selected[0]



@click.command(help='List available checkpoints.')
def list():
    db = read_checkpoint_db()

    if not db['checkpoints']:
        click.echo('No checkpoints available.')
        return

    template = '| {:>30} | {:>12} | {:>8} | {:>8} | {:>14} |'

    header = template.format(
        'Name', 'Alias', 'Eval.', 'Perf.', 'Status'
    )
    click.echo('=' * len(header))
    click.echo(header)

    is_first = True
    for key, group in groupby(db['checkpoints'], key=lambda x: x['task']):
        label = LABELS_BY_TASK.get(key, '')
        click.echo(('=' if is_first else '-') * len(header))
        click.echo(f'| {label:<{len(header) - 4}} |')
        click.echo('-' * len(header))
        is_first = False

        for checkpoint in group:
            line = template.format(
                f"{checkpoint['name']} ({checkpoint['id']})",
                checkpoint['alias'],
                (
                    ('* ' if checkpoint['evaluation']['is_reported'] else '')
                    + f"{checkpoint['evaluation']['value']:.2f}"
                ),
                f"{checkpoint['performance']:.2f}",
                checkpoint['status'],
            )
            click.echo(line)

    click.echo('=' * len(header))


@click.command(help='Display detailed information on checkpoint.')
@click.argument('checkpoint_id')
def info(checkpoint_id):
    db = read_checkpoint_db()

    checkpoint = get_checkpoint(db, checkpoint_id)
    if not checkpoint:
        click.echo(
            "Checkpoint '{}' not found in index.".format(checkpoint_id)
        )
        return

    click.echo(
        f"{checkpoint['name']} ({checkpoint['id']}, {checkpoint['alias']})"
    )

    if checkpoint['description']:
        click.echo(f" > {checkpoint['description']}")

    click.echo()

    click.echo(f"Task: {LABELS_BY_TASK.get(checkpoint['task'], '')}")
    click.echo('Evaluation information: {:.3f} {}{}'.format(
        checkpoint['evaluation']['value'],
        checkpoint['evaluation']['metric'],
        ' (self-reported)' if checkpoint['evaluation']['is_reported'] else ''
    ))
    click.echo(
        f"Computational performance: {checkpoint['performance']:.2f} units"
    )

    click.echo()

    click.echo(f"Upstream URL: {checkpoint['url']}")
    if checkpoint['local_path']:
        click.echo(f"Status: DOWNLOADED (at `{checkpoint['local_path']}`)")
    else:
        click.echo('Status: NOT_DOWNLOADED')



@click.group(name='checkpoint', help='Checkpoint management commands.')
def checkpoint_cmd():
    pass


checkpoint_cmd.add_command(info)
checkpoint_cmd.add_command(list)
