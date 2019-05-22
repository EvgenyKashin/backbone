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

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
