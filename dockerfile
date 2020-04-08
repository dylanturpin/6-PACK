FROM nvidia/cudagl:10.0-devel-ubuntu16.04
# apt installable dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y &&\
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  libsm6 \
  libxext6 \
  libxrender-dev \
  wget
# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ./anaconda.sh &&\
  /bin/bash ./anaconda.sh -b -p /opt/conda &&\
  /opt/conda/bin/conda init bash &&\
  rm ./anaconda.sh
RUN /opt/conda/bin/conda create -n 6pack python=3.7
RUN /opt/conda/bin/conda install -n 6pack -c conda-forge scipy pip
RUN /opt/conda/bin/conda run -n 6pack pip install --upgrade pip
RUN /opt/conda/bin/conda run -n 6pack pip install \
  torch==0.4.1.post2 \
  torchvision==0.2.1 \
  pillow==6.1 \
  opencv-python
RUN mkdir /pkgs &&\
  cd /pkgs &&\
  git clone https://github.com/dylanturpin/hydra &&\
  cd /pkgs/hydra &&\
  /opt/conda/bin/conda run -n 6pack pip install -e .
