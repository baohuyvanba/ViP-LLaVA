FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

# Ubuntu Mirror server
ARG UBUNTU_MIRROR=archive.ubuntu.com/ubuntu
#docker build --build-arg UBUNTU_MIRROR=us.archive.ubuntu.com/ubuntu .

# Change mirror and Force apt only run IPv4
RUN sed -i "s|http://archive.ubuntu.com/ubuntu|http://$UBUNTU_MIRROR|g" /etc/apt/sources.list
# echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# Update cache, turn on universe, install system packages
RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -yq \
        ffmpeg \
        nvidia-cuda-toolkit \
        dkms \
        build-essential \
        jq \
        jp \
        tree \
        tldr \
        git \
        wget \
        curl \
        ca-certificates

# Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -yq git-lfs && \
    git lfs install

############################################
# Setup user
############################################
USER vscode
WORKDIR /home/vscode

# Miniconda
ENV CONDA_DIR=/home/vscode/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p "$CONDA_DIR" && \
    rm miniconda.sh && \
    conda init --all

# Initial Conda Environment
RUN conda create -n vip-llava python=3.10 -y && \
    conda clean -afy

# Clone Repo and Install Requirements
RUN git clone https://github.com/baohuyvanba/ViP-LLaVA.git /home/vscode/ViP-LLaVA && \
    conda run -n vip-llava pip install --upgrade pip && \
    conda run -n vip-llava pip install -e /home/vscode/ViP-LLaVA && \
    conda run -n vip-llava pip install -e "/home/vscode/ViP-LLaVA[train]" && \
    conda run -n vip-llava pip install huggingface_hub

RUN conda run -n vip-llava pip install accelerate==0.32.0 && \
    conda run -n vip-llava pip install peft==0.10.0

# STAGE 1: Download LLaVA Pretrain Data
# RUN mkdir -p /home/vscode/ViP-LLaVA/playground/data/LLaVA-Pretrain && \
#     cd /home/vscode/ViP-LLaVA/playground/data/LLaVA-Pretrain && \
#     wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json && \
#     wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json && \
#     wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
    # unzip images.zip -d images && \
    # rm images.zip

#STAGE 2
# COPY vip-llava_stage2_mix_coco.json /home/vscode/ViP-LLaVA/playground/data
# # RUN cd /home/vscode/ViP-LLaVA/playground/data && \
# #     wget -c http://images.cocodataset.org/zips/train2017.zip