import click
import importlib
import os
import requests
import shutil
import sys
import tempfile

from itertools import groupby
from pathlib import Path


__all__ = [
    'get_terran_home',
    'get_class_for_checkpoint',
    'get_checkpoint_path',
]


DEFAULT_TERRAN_HOME = Path('~/.terran')
CHECKPOINT_PATH = 'checkpoints'

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
        'class': 'terran.face.detection.retinaface.RetinaFace',
        'alias': 'gpu-realtime',
        'default': True,

        'performance': 1.0,
        'evaluation': {
            'value': 0.76,
            'metric': 'mAP',
            'is_reported': False,
        },

        'url': (
            'https://github.com/nagitsu/terran/releases/download/0.0.1/'
            'retinaface-mnet.pth'
        )
    },
    {
        'id': 'd206e4b0',
        'name': 'ArcFace',
        'description': (
            'ArcFace with Resnet 100 backbone.'
        ),

        'task': 'face-recognition',
        'class': 'terran.face.recognition.arcface.ArcFace',
        'alias': 'gpu-realtime',
        'default': True,

        'performance': 0.9,
        'evaluation': {
            'value': 0.80,
            'metric': 'accuracy',
            'is_reported': False,
        },

        'url': (
            'https://github.com/nagitsu/terran/releases/download/0.0.1/'
            'arcface-resnet100.pth'
        )
    },
    {
        'id': '11a769ad',
        'name': 'OpenPose',
        'description': (
            'OpenPose with VGG backend, 2017 version. Has some modifications, '
            'improving computational efficiency by giving up mAP.'
        ),

        'task': 'pose-estimation',
        'class': 'terran.pose.openpose.OpenPose',
        'alias': 'gpu-realtime',
        'default': True,

        'performance': 1.8,
        'evaluation': {
            'value': 0.65,
            'metric': 'mAP',
            'is_reported': True,
        },

        'url': (
            'https://github.com/nagitsu/terran/releases/download/0.0.1/'
            'openpose-body.pth'
        )
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


def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    # Get the downloaded checkpoints by searching through the filesystem.
    local_checkpoints = set(
        path.stem for path in get_checkpoints_directory().glob('*.pth')
    )

    checkpoints = [
        {
            'status': (
                'DOWNLOADED' if checkpoint['id'] in local_checkpoints
                else 'NOT_DOWNLOADED'
            ),
            'local_path': (
                get_checkpoints_directory() / f"{checkpoint['id']}.pth"
                if checkpoint['id'] in local_checkpoints else None
            ),
            **checkpoint
        }
        for checkpoint in CHECKPOINTS
    ]

    return {
        'checkpoints': checkpoints
    }


def get_checkpoint(db, id_or_alias):
    """Returns checkpoint entry in `db` indicated by `id_or_alias`.

    Parameters
    ----------
    id_or_alias : str or tuple of str
        Either the ID of the checkpoint, or a tuple of the form (task_name,
        alias) that uniquely points to a checkpoint.

        In the case of the latter, specifying `alias` as `None` will default to
        the entry marked as `default = True` in the index.

    Returns
    -------
    Dict
        Checkpoint data contained in the database.

    """
    if isinstance(id_or_alias, tuple):
        task_name, alias = id_or_alias
        selected = [
            c for c in db['checkpoints']
            if c['task'] == task_name and (
                c['alias'] == alias if alias is not None else c['default']
            )
        ]
    else:
        selected = [c for c in db['checkpoints'] if c['id'] == id_or_alias]

    if len(selected) < 1:
        return None

    if len(selected) > 1:
        click.echo(
            f"Multiple checkpoints found for '{id_or_alias}' "
            f"({len(selected)}). Returning first."
        )

    return selected[0]


def get_class_for_checkpoint(task_name, alias):
    """Returns the model class for the given ID or alias.

    Parameters
    ----------
    task_name : str
        Name of the task to get the class for.
    alias : str or `None`
        Alias for the checkpoint, within `task_name`, to get the class for.
        If `None`, will use the first entry that specifies `default = True`.

    Returns
    -------
    class
        The class of the model associated to the checkpoint.

    Raises
    ------
    ValueError
        If checkpoint is not found.
    ImportError
        If the class is somehow misspecified or not present.

    """
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, (task_name, alias))

    if not checkpoint:
        # No checkpoint found.
        raise ValueError('Checkpoint not found.')

    module_path, class_name = checkpoint['class'].rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module_path), class_name)


def get_checkpoint_by_class(db, class_path):
    """Returns checkpoint entry in `db` indicated by `class_path`.

    Parameters
    ----------
    class_path : str
        Fully specified path to class (e.g. `terran.pose.openpose.OpenPose`)
        of the model to get the checkpoint for.

    Returns
    -------
    Dict
        Checkpoint data contained in the database.

    """
    selected = [c for c in db['checkpoints'] if c['class'] == class_path]

    if len(selected) < 1:
        return None

    if len(selected) > 1:
        click.echo(
            f"Multiple checkpoints found for '{class_path}' "
            f"({len(selected)}). Returning first."
        )

    return selected[0]


def get_checkpoint_path(model_class_path, prompt=True):
    """Returns the local path to the model's weights.

    Goes through the list of checkpoints and returns the local path to the
    weights of the modell specified by `model_class`. If the weights are not
    downloaded, downloads them first.

    Parameters
    ----------
    model_class_path : str
        Fully specified path to class (e.g. `terran.pose.openpose.OpenPose`)
        of the model to get the checkpoint for.
    prompt : boolean
        If `True` and the checkpoint is not yet downloaded, prompt to download.

    Returns
    -------
    pathlib.Path
        Path to the `.pth` file containing the weights for the model.

    Raises
    ------
    ValueError
        If checkpoint is not found or is found but not downloaded, either due
        to aborting the prompt or disabling it in the first place.

    """
    db = read_checkpoint_db()
    checkpoint = get_checkpoint_by_class(db, model_class_path)
    can_prompt = sys.stdout.isatty()

    if not checkpoint:
        # No checkpoint found.
        raise ValueError('Checkpoint not found.')

    if checkpoint['status'] == 'NOT_DOWNLOADED':
        if prompt and can_prompt:
            # Checkpoint hasn't been downloaded yet. Prompt for downloading it
            # before continuing.
            try:
                click.confirm(
                    'Checkpoint not present locally. Want to download it?', abort=True
                )
            except Exception:
                click.echo('Checkpoint not present locally. Downloading it')

        download_remote_checkpoint(db, checkpoint)

    # `local_path` is now present in the checkpoint, as
    # `download_remote_checkpoint` will have modified it if it was just
    # downloaded.
    return checkpoint['local_path']


def download_remote_checkpoint(db, checkpoint):
    # Check if the checkpoint is already downloaded. If it is, this function
    # shouldn't have been called, so something is inconsistent.
    if checkpoint['local_path'] and checkpoint['local_path'].exists():
        click.echo(
            f"Checkpoint file already present at {checkpoint['local_path']}. '"
            f"If you're running into any issues, try issuing a `terran "
            f"checkpoint delete {checkpoint['id']}` trying attempting again."
        )
        return

    file_name = f"{checkpoint['id']}.pth"

    # Create a temporary directory to download the files into.
    tempdir = tempfile.mkdtemp()
    path = Path(tempdir) / file_name

    # Start the actual file download.
    response = requests.get(checkpoint['url'], stream=True)

    if response.status_code != 200:
        raise ValueError(f'Invalid checkpoint URL {checkpoint["url"]}')

    length = int(response.headers.get('Content-Length'))
    chunk_size = 16 * 1024
    progressbar = click.progressbar(
        response.iter_content(chunk_size=chunk_size),
        length=length / chunk_size, label='Downloading checkpoint...',
    )

    with open(path, 'wb') as f:
        with progressbar as content:
            for chunk in content:
                f.write(chunk)

    # Move the file to the checkpoints directory.
    new_path = get_checkpoints_directory() / file_name
    shutil.move(path, new_path)

    # Update the checkpoint dict information.
    checkpoint['status'] = 'DOWNLOADED'
    checkpoint['local_path'] = new_path

    # And finally make sure to delete the temp dir.
    shutil.rmtree(tempdir)

    click.echo("Checkpoint downloaded successfully.")


@click.command(name='list', help='List available checkpoints.')
def list_cmd():
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
                '',
                '',
                # (
                #     ('* ' if checkpoint['evaluation']['is_reported'] else '')
                #     + f"{checkpoint['evaluation']['value']:.2f}"
                # ),
                # f"{checkpoint['performance']:.2f}",
                checkpoint['status'],
            )
            click.echo(line)

    click.echo('=' * len(header))


@click.command(name='info', help='Display detailed information on checkpoint.')
@click.argument('checkpoint_id')
def info_cmd(checkpoint_id):
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
    click.echo(f"Class: `{checkpoint['class']}`")
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


@click.command(
    name='delete', help='Delete local files associated to a checkpoint.'
)
@click.argument('checkpoint_id')
def delete_cmd(checkpoint_id):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, checkpoint_id)
    if not checkpoint:
        click.echo(f"Checkpoint `{checkpoint_id}` not found in index.")
        return

    if checkpoint['status'] == 'NOT_DOWNLOADED':
        click.echo("Checkpoint isn't downloaded. Nothing to delete.")
        return

    # Delete the files associated to checkpoint.
    checkpoint['local_path'].unlink()

    click.echo(f"Checkpoint `{checkpoint['id']}` deleted successfully.")


@click.command(name='download', help='Download a remote checkpoint.')
@click.argument('checkpoint_id')
def download_cmd(checkpoint_id):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, checkpoint_id)
    if not checkpoint:
        click.echo(f"Checkpoint `{checkpoint_id}` not found in index.")
        return

    if checkpoint['status'] != 'NOT_DOWNLOADED':
        click.echo("Checkpoint is already downloaded.")
        return

    download_remote_checkpoint(db, checkpoint)


@click.group(name='checkpoint', help='Checkpoint management commands.')
def checkpoint_cmd():
    pass


checkpoint_cmd.add_command(delete_cmd)
checkpoint_cmd.add_command(download_cmd)
checkpoint_cmd.add_command(info_cmd)
checkpoint_cmd.add_command(list_cmd)
