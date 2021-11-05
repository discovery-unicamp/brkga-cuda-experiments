#!/bin/bash

set -e

cmake -DCMAKE_BUILD_TYPE=Debug -Bcmake-build-release .
cmake --build cmake-build-release -- -j 6
time ./cmake-build-release/tsp config/tsp instances/tsp/a280.tsp 1000
