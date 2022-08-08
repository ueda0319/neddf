ARG BASE_IMAGE=nvidia/cuda:11.1.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=neddf
ARG USER_NAME=challenger
ARG PYTHON_VERSION=3.8
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

# Following is needed to install python 3.7
RUN apt update && \
    apt install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y \
    git \
    curl \
    make \
    ssh \
    openssh-client \
    libgl1-mesa-dev \
    python${PYTHON_VERSION} python3-pip python-is-python3
    
# Following is needed to swtich default python3 version
# For detail, please check following link https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    python3 -m pip install --upgrade pip 'setuptools==59.5.0' && \
    python3 -m pip install --pre poetry

# Add user. Without this, following process is executed as admin. 
RUN useradd -ms /bin/sh ${USER_NAME}
USER ${USER_NAME}

WORKDIR ${APPLICATION_DIRECTORY}
# Create virtual environments inside of project.
RUN poetry config virtualenvs.in-project true
