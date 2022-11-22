import subprocess
from pathlib import Path

cellpose_dir = Path('data/cellpose/')

models_dir = Path('weights')

cellpose_config = {
    'model_to_load': 'cyto2',
    'number_of_epochs': 100,
    'initial_learning_rate': 0.0002,
    'diameter': 16,
    'batch_size': 64,
}


def train():
    train_dir = cellpose_dir / f'fold_0' / 'train'
    val_dir = cellpose_dir / f'fold_0' / 'val'

    (train_dir / 'models').mkdir(exist_ok=True, parents=True)

    # training cellpose model
    subprocess.run([
        'python', '-m', 'cellpose', '--train', '--use_gpu', '--fast_mode',
        '--dir', train_dir,
        '--test_dir', val_dir,
        '--pretrained_model', cellpose_config['model_to_load'],
        '--chan', str(0),
        '--chan2', str(0),
        '--n_epochs', str(cellpose_config['number_of_epochs']),
        '--learning_rate', str(cellpose_config['initial_learning_rate']),
        '--batch_size', str(cellpose_config['batch_size']),
        '--img_filter', 'img',
        '--mask_filter', 'masks',
        '--diameter', str(cellpose_config['diameter']),
        '--verbose'
    ])

    print("finished!")


if __name__ == '__main__':
    train()
