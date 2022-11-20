import os
import shutil
import subprocess
from pathlib import Path

cellpose_dir = Path('data/cellpose/')

models_dir = Path('weights')

cellpose_config = {
    'model_to_load': 'cyto2',

    'number_of_epochs': 50,
    'initial_learning_rate': 0.0002,
    'diameter': 16,
    'batch_size': 8,
}


# python -m cellpose --train --use_gpu --fast_mode \
#         --dir "$train_folder" --test_dir "$test_folder" \
#         --pretrained_model $model_to_load \
#         --chan $Training_channel --chan2 $Second_training_channel \
#         --n_epochs $number_of_epochs \
#         --learning_rate $initial_learning_rate \
#         --batch_size $batch_size \
#         --img_filter img \
#         --mask_filter masks \
#         --diameter $diameter

def train():
    train_dir = str(cellpose_dir / f'fold_0' / 'train')
    val_dir = str(cellpose_dir / f'fold_0' / 'val')
    # training cellpose model
    subprocess.run([
        'python', '-m', 'cellpose', '--train', '--use_gpu', '--fast_mode',
        '--dir', train_dir,
        '--test_dir', val_dir,
        '--pretrained_model', cellpose_config['model_to_load'],
        '--chan', 0,
        '--chan2', 0,
        '--n_epochs', cellpose_config['number_of_epochs'],
        '--learning_rate', cellpose_config['initial_learning_rate'],
        '--batch_size', cellpose_config['batch_size'],
    ])

    shutil.copytree(os.path.join(train_dir, 'models'), models_dir)

    print("finished!")


if __name__ == '__main__':
    train()
