mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
project_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)
device := 0

run: .setup
	docker run -it --env DEVICE=$(device) -v $(project_path)/:/experiment/ --rm --gpus device=$(device) brkga

.setup: experiments/Dockerfile experiments/requirements.txt
	rm -f core
	docker build --no-cache --build-arg CUDA_VERSION=$(cuda_version) -t brkga -f experiments/Dockerfile .
	docker rmi --force $(docker images -f "dangling=true" -q) || echo "No images to remove"
	echo "Setup on: $$(date)" >.setup

tuning: .tuning-setup
	docker run -it --env DEVICE=$(device) -v $(project_path)/:/experiment/ --rm --gpus device=$(device) tuning

.tuning-setup: experiments/Dockerfile.tuning experiments/requirements.txt
	rm -f core
	docker build --no-cache --build-arg CUDA_VERSION=$(cuda_version) -t tuning -f experiments/Dockerfile.tuning .
	docker rmi --force $(docker images -f "dangling=true" -q) || echo "No images to remove"
	echo "Setup on: $$(date)" >.tuning-setup

.PHONY: open-terminal
open-terminal:
	docker run -it -v $(project_path)/:/experiment/ --rm ubuntu

.PHONY: fix-git
fix-git: # Rule created due the errors on dl-1
	find .git/objects/ -size 0 -exec rm -f {} \;
	git fetch origin

.PHONY: clean
clean:
	docker run -it -v $(project_path)/:/experiment/ --rm ubuntu /bin/bash -c 'cd experiment; rm -f .setup .tuning-setup core; rm -rf build*'
	docker rmi $$(docker images 'brkga' -a -q) || exit 0
	docker rmi $$(docker images 'tuning' -a -q) || exit 0
