import click

from terran.checkpoint import checkpoint_cmd


@click.group()
def cli():
    pass


cli.add_command(checkpoint_cmd)
