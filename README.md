# Readme

## directory structure

- `./src` source code
- `./scripts` entry points, try `python scripts/train.py`
- `./conf` configuration for each entry point, e.g. `./conf/train/config.yaml` configures train script
- `./docker` docker and singularity files to build container with all requirements
- `./.circleci` circleci build config: builds docker image and singularity sandbox, tars sandbox and pushes to vector
