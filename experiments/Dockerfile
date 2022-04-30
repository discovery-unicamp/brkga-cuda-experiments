FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 AS build

WORKDIR /brkga

ENV DEBIAN_FRONTEND=noninteractive

# install the required tools
RUN apt-get update \
    && apt-get install --no-install-recommends -y git pciutils lshw wget libssl-dev
    && apt-get install --no-install-recommends -y libgomp1 python3 python3-pip \
    && pip3 install -r requirements.txt \
    && wget https://github.com/Kitware/CMake/releases/download/v3.20.6/cmake-3.20.6.tar.gz \
    && tar -xf cmake*.tar.gz \
    && cd cmake* \
    && ./configure --parallel=6 \
    && make -j6 \
    && make install \
    && cmake --version \
    && apt-get remove -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# get system info
# COPY .git/ .git/
# RUN printf %"s\n" \
#     "commit: $(git log --format="%H" -n 1)" \
#     "system: $(uname -v)" \
#     "cpu: $(cat /proc/cpuinfo | grep "model name" | uniq | cut -d" " -f 3-)" \
#     "host-memory: $(grep MemTotal /proc/meminfo | awk '{print $2 / 1024}')MiB" \
#     "gpu: $(lspci | grep " VGA " | cut -d" " -f 5-)" \
#     "gpu-memory: $(lshw -C display | grep product | cut -d":" -f2-)" \
#     "nvcc: $(nvcc --version | grep "release" | grep -o "V.*")" \
#     "g++: $(g++ --version | grep "g++")" >info.txt

# compile the applications
#! FIXME move the other sources to a specific folder
#! FIXME use wget or git clone to fetch other sources
# COPY GPU-BRKGA/ GPU-BRKGA/
# COPY toso-brkga-api/brkgaAPI/ toso-brkga-api/brkgaAPI/
# COPY brkga-cuda-itor/ brkga-cuda-itor/
# COPY applications/ applications/
# RUN cmake -DCMAKE_BUILD_TYPE=release -Bbuild applications/ \
#     && cmake --build build --target brkga-optimizer -j6

# copy the data and the scripts
# COPY experiments/instances/ ./instances/
# COPY experiments/*.py ./
# COPY --from=build /brkga/info.txt ./info.txt
# COPY --from=build /brkga/build/brkga-optimizer ./brkga-cuda

# set the experiment to run
CMD [ "python3", "experiment.py" ]
