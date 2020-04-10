FROM nvidia/cudagl:10.0-devel-ubuntu16.04
# apt installable dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  libsm6 \
  libstdc++6 \
  libxext6 \
  libxrender-dev \
  software-properties-common \
  wget
# update libstdc++6
RUN add-apt-repository ppa:ubuntu-toolchain-r/test &&\
  DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y libstdc++6

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ./anaconda.sh &&\
  /bin/bash ./anaconda.sh -b -p /opt/conda &&\
  /opt/conda/bin/conda init bash &&\
  rm ./anaconda.sh
RUN /opt/conda/bin/conda create -n 6pack python=3.7
RUN /opt/conda/bin/conda install -n 6pack -c conda-forge scipy pip
RUN /opt/conda/bin/conda run -n 6pack pip install --upgrade pip
RUN /opt/conda/bin/conda run -n 6pack pip install \
  cffi \
  torch \
  torchvision \
  opencv-python \
  pillow \
  pyquaternion \
  wandb
RUN mkdir /pkgs &&\
  cd /pkgs &&\
  git clone https://github.com/dylanturpin/hydra &&\
  cd /pkgs/hydra &&\
  /opt/conda/bin/conda run -n 6pack pip install -e .
RUN cd /pkgs &&\
  git clone https://github.com/chrischoy/gesvd.git &&\
  cd /pkgs/gesvd &&\
  /opt/conda/bin/conda run -n 6pack python setup.py install
