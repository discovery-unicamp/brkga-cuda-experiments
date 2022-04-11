FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 AS build

WORKDIR /brkga

ENV DEBIAN_FRONTEND=noninteractive

# install the required tools
RUN apt-get update \
    && apt-get install --no-install-recommends -y gcc g++ make wget libssl-dev \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.6/cmake-3.20.6.tar.gz \
    && tar -xf cmake*.tar.gz \
    && cd cmake* \
    && ./configure --parallel=6 \
    && make -j6 \
    && make install \
    && cmake --version

# compile the applications
#! FIXME move the other sources to a specific folder
#! FIXME use wget or git clone to fetch other sources
COPY applications/ applications/
COPY brkga-cuda-itor/ brkga-cuda-itor/
COPY GPU-BRKGA/ GPU-BRKGA/
RUN cmake -DCMAKE_BUILD_TYPE=release -Bbuild applications/ \
    && cmake --build build --target brkga-cvrp -j6

################################################################################
FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04 AS main

WORKDIR main

# install dependencies
COPY experiments/requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 python3 python3-pip \
    && pip3 install -r requirements.txt \
    && apt-get remove -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# copy the executables
COPY --from=build /brkga/build/cvrp/brkga-cvrp ./brkga-cuda
COPY experiments/ ./

# set the experiment to run
CMD [ "python3", "experiment.py" ]
