FROM nvidia/cuda:10.2-runtime

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update


# Run python installation
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade

# Install conda 
RUN apt-get install wget
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Clone the repo
RUN ls
RUN git clone https://github.com/oljikeboost/Tracking.git
RUN ls
RUN cd Tracking
RUN ls
RUN git clone https://github.com/jinfagang/DCNv2_latest
RUN ls
RUN conda env create -f Tracking/env.yml
# RUN conda init bash 
SHELL ["conda", "run", "-n", "FairMOT", "conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch"]


RUN cd DCNv2_latest

RUN ./make.sh
RUN cd .. 

CMD ["python", "/src/inference.py"]