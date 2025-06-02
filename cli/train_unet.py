import logging
import click
from pathlib import Path
from fontcap_model import train_unet

"""
Click command for training the U-Net. Usage (from project root with venv):
python -m cli.train_unet -dr "data/fonts" -ep 2 -bs 32 -lr 0.001 -chkdir "./something" -chki 1 -pi 1
"""

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='[%(asctime)s] %(levelname)s: %(message)s',
)


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
@click.option('--data_root', '-dr', type=click.Path(exists=True), required=True, help='Path to data')
@click.option('--epochs', '-ep', type=int, required=True, help='Number of training epochs')
@click.option('--batch_size', '-bs', type=int, required=True, help='Batch size per epoch')
@click.option('--learning_rate', '-lr', type=float, required=True, help='Learning rate')
@click.option('--checkpoint_dir', '-chkdir', type=click.Path(), required=False,
              default=Path("./checkpoints"), help='Number of training epochs')
@click.option('--checkpoint_interval', '-chki', type=int, required=False,
              default=20, help='Save the model parameters every x epochs')
@click.option('--plot_interval', '-pi', type=int, required=False,
              default=20, help='Save example model outputs every x epochs')
@click.option('--start_state', '-st', type=click.Path(exists=True), required=False,
              help='Starting parameters file (stored in checkpoints dir')
@click.option('--resume_loss', '-rl', is_flag=True, help='Resume loss curve')
def run(data_root, epochs, batch_size, learning_rate, checkpoint_dir, checkpoint_interval, plot_interval, start_state,
        resume_loss, verbose):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    train_unet(
        data_root=data_root,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        plot_interval=plot_interval,
        state_dict_name=start_state,
        resume_loss=resume_loss)
    return


if __name__ == '__main__':
    run()
