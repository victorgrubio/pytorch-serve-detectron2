FROM pytorch/torchserve:0.1.1-cpu

ENV DEBIAN_FRONTEND noninteractive
USER root
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/model-server

ENV PATH="/home/model-server/.local/bin:${PATH}"
# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3 python-dev python3-dev \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
USER model-server
# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/model-server
RUN git clone https://github.com/facebookresearch/detectron2.git
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]