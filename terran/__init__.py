import click
import torch


default_device = (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)


@click.command()
def cli():
    click.echo('Why are you looking so far back?')
