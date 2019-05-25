FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    htop \
    libturbojpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    rsync

RUN pip install back==0.0.3 \
                matplotlib==3.1.0 \
                torchsummary==1.5.1 \
                tb-nightly==1.14.0a20190522 \
                jupyter==1.0.0

