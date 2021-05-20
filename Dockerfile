ARG IMAGE_NAME
FROM nvidia/cuda:10.2-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.2 \
libcublas-dev=10.2.2.89-1 \
&& \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# CT: 4/17


# Install Miniconda
# RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
# RUN conda install -y -c pytorch \
#     cudatoolkit=10.0 \

    #"pytorch=1.6.0=py3.6_cuda10.2.130_cudnn7.6.5_0" \
    #"torchvision=0.7.0=py36_cu102" \
RUN conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch \
    && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3 \
 && conda clean -ya
 
 
### Install mmdetection
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && sudo  apt-get clean \
    && sudo  rm -rf /var/lib/apt/lists/*

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/oljikeboost/mmdet /home/user/mmdetection
WORKDIR /home/user/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install mmcv from the source 
WORKDIR /home/user
RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /home/user/mmcv
RUN git checkout f4de390b3c808fe20b1d36d783cac5d7887b41e9
RUN MMCV_WITH_OPS=1 pip install -e .


### Additional packages
RUN pip install opencv-python && pip install cython-bbox && pip install sklearn && pip install numba && pip install yacs && \
    pip install lap

### insert some random VAR to break cahche
ARG INCUBATOR_VER=unknown3

### Install nano
RUN sudo apt-get update && sudo apt-get install nano

### Clone the Tracking Git 
RUN git clone https://github.com/oljikeboost/Tracking.git /home/user/Tracking/

### Run cython build
WORKDIR /home/user/Tracking/src/lib/tracker 
RUN python setup.py build_ext --inplace

### install DCN
RUN git clone https://github.com/oljikeboost/DCNv2.git /home/user/Tracking/DCNv2_latest/
WORKDIR /home/user/Tracking/DCNv2_latest
RUN ./make.sh


### Download all weights to docker internal directory
RUN mkdir /home/user/weights
WORKDIR /home/user/weights
RUN wget -q https://boost-operators-data.s3.us-east-2.amazonaws.com/tracker_weights/epoch_90.pth && \
    wget -q https://boost-operators-data.s3.us-east-2.amazonaws.com/tracker_weights/model-best.pth && \
    wget -q https://boost-operators-data.s3.us-east-2.amazonaws.com/tracker_weights/yolov3_d53_320_273e_jersey_smallres.py && \
    wget -q https://boost-operators-data.s3.us-east-2.amazonaws.com/tracker_weights/default_runtime.py && \
    wget -q https://boost-operators-data.s3.us-east-2.amazonaws.com/tracker_weights/model_30.pth

WORKDIR /home/user/Tracking
RUN chmod +x inference.sh


# Set the default command to python3
ENTRYPOINT ["./inference.sh"]