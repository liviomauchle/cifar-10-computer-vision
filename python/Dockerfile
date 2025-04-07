FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

ARG DOCKER_USER
ARG DOCKER_UID
ARG DOCKER_GID

# Prevent interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    xvfb \
    x11-apps \
    vim \
    mesa-utils \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    python3-tk \
    python3-venv \
    pyqt5-dev-tools \
    python3-pyqt5.qtsvg \
    python3-pyqt5 \
    python3-pyqt5.qtwebengine \
    libxcb-xinerama0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Install the required Python packages
RUN pip3 install --no-cache-dir \
    h5py \
    matplotlib \
    opencv-python \
    pytest \
    spyder \
    tensorflow_datasets

RUN groupadd --gid $DOCKER_GID $DOCKER_USER && \
    useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER

RUN mkdir -p /home/$DOCKER_USER/.config

WORKDIR /home/$DOCKER_USER

RUN spyder --reset

RUN chown -R $DOCKER_USER:$DOCKER_USER /home/$DOCKER_USER
RUN chown -R $DOCKER_USER:$DOCKER_USER /home/$DOCKER_USER/.config

RUN usermod -aG video $DOCKER_USER

USER $DOCKER_USER
