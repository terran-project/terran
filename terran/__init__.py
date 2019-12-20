import click
import torch

from terran.checkpoint import checkpoint_cmd


default_device = (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)


@click.group()
def cli():
    pass


cli.add_command(checkpoint_cmd)
