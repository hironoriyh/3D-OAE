#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER root
WORKDIR /tmp

ARG PYTHON_VERSION=3.9
ARG WITH_TORCHVISION=1

RUN conda create -n md python=3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         ca-certificates \
	 apt-utils apt-transport-https pkg-config \
	 software-properties-common \
         libjpeg-dev \
	 libopencv-dev \
         libpng-dev \
	 libopenexr-dev \
	 libegl1-mesa-dev \
     blender \ 
     gfortran lsof \
     # Blender and openSCAD are soft dependencies used in Trimesh for boolean operations with subprocess
	 vim tmux git wget curl && \
     rm -rf /var/lib/apt/lists/*


SHELL ["conda", "run", "-n", "md", "/bin/bash", "-c"]

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath 

RUN conda run -n md && pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113


#RUN conda run -n md  && pip install --upgrade pip


RUN conda run -n md  && pip install ninja imageio PyOpenGL glfw xatlas gdown

RUN conda run -n md  && pip install git+https://github.com/NVlabs/nvdiffrast/


RUN conda run -n md  && pip install   einops  scipy matplotlib
#uUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 

RUN conda run -n md  && pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html 

#RUN    pip install clip-by-openai
RUN conda run -n md  && pip install tqdm usd-core ipdb

RUN conda run -n md  && pip install "git+https://github.com/facebookresearch/pytorch3d.git"

COPY requirements.txt /tmp/

RUN cd /tmp && \
    pip install -r requirements.txt 

#RUN apt-get update && apt-get install -y --no-install-recommends \ 
# apt-get install    libegl1-mesa-dev \
   # rm -rf /var/lib/apt/lists/*
#RUN  pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
#RUN imageio_download_bin freeimage



