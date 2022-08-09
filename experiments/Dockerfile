ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu20.04 AS build

WORKDIR /brkga

ENV DEBIAN_FRONTEND=noninteractive

# install the required tools
COPY experiments/requirements.txt requirements.txt
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update \
    && apt-get install -y wget \
    && apt-get install --no-install-recommends -y libssl-dev \
    && apt-get install --no-install-recommends -y git pciutils lshw \
    && apt-get install --no-install-recommends -y libgomp1 \
    && apt-get install --no-install-recommends -y python3 python3-pip \
    && pip3 install -r requirements.txt \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.6/cmake-3.20.6.tar.gz \
    && tar -xf cmake*.tar.gz \
    && cd cmake* \
    && ./configure --parallel=6 \
    && make -j6 \
    && make install \
    && cmake --version \
    && apt-get remove -y python3-pip wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set the experiment to run
CMD [ "python3", "experiments/experiment.py" ]
