import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))


import hydra
from hydra import slurm_utils

@hydra.main(config_path='../conf/train/config.yaml')
def main(config):
    slurm_utils.symlink_hydra(config, os.getcwd())

if __name__ == "__main__":
    main()
