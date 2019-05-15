FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN apt install wget unzip htop
RUN pip install pandas jupyter pillow==5.4.1 tqdm
