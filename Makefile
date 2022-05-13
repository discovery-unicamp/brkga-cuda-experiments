mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
project_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)
device := 0

run: .setup
	docker run --env DEVICE=$(device) -v $(project_path)/:/brkga/ --rm --gpus device=$(device) brkga

.setup: experiments/Dockerfile experiments/requirements.txt
	docker build --no-cache --build-arg CUDA_VERSION=$(cuda_version) -t brkga -f experiments/Dockerfile .
	echo "Setup on: $$(date)" >.setup

.PHONY: open-terminal
open-terminal:
	docker run -v $(project_path)/:/brkga/ --rm -it ubuntu

.PHONY: clean
clean:
	rm -f .setup core
	rm -rf build-*
