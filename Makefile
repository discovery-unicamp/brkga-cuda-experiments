mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
project_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)
device := 0

DOCKER_RUN = docker run -it -u $$(id -u):$$(id -g) -v $(project_path)/:/experiment/ --rm


run: .setup
	git log --format="%H" -n 1 >.commit
	$(DOCKER_RUN) --env DEVICE=$(device) --gpus device=$(device) brkga

.setup: experiments/Dockerfile experiments/requirements.txt
	docker build --build-arg CUDA_VERSION=$(cuda_version) -t brkga -f experiments/Dockerfile .
	docker rmi --force $(docker images -f "dangling=true" -q) || echo "No images to remove"
	echo "Setup on: $$(date)" >.setup

tuning: .setup-tuning
	$(DOCKER_RUN) --env DEVICE=$(device) --gpus device=$(device) tuning

.setup-tuning: experiments/Dockerfile.tuning experiments/requirements.txt
	docker build --build-arg CUDA_VERSION=$(cuda_version) -t tuning -f experiments/Dockerfile.tuning .
	docker rmi --force $(docker images -f "dangling=true" -q) || echo "No images to remove"
	echo "Setup on: $$(date)" >.setup-tuning

.PHONY: open-terminal
open-terminal:
	$(DOCKER_RUN) ubuntu

.PHONY: open-nvidia
open-nvidia: .setup-nvidia
	$(DOCKER_RUN) nvidia

.setup-nvidia: experiments/Dockerfile.nvidia experiments/requirements.txt
	docker build --build-arg CUDA_VERSION=$(cuda_version) -t nvidia -f experiments/Dockerfile.nvidia .
	docker rmi --force $(docker images -f "dangling=true" -q) || echo "No images to remove"
	echo "Setup on: $$(date)" >.setup-nvidia

.PHONY: update-submodules
update-submodules:
	git submodule update --init --recursive

.PHONY: fix-git
fix-git: # Rule created due the errors on dl-1
	find .git/objects/ -size 0 -exec rm -f {} \;
	git fetch origin

.PHONY: clean
clean:
	rm -f .setup* core
	rm -rf build*'
	docker rmi $$(docker images 'brkga' -a -q) || (echo "Ignoring 'docker rmi'" && exit 0)
	docker rmi $$(docker images 'tuning' -a -q) || (echo "Ignoring 'docker rmi'" && exit 0)
	docker rmi $$(docker images 'nvidia' -a -q) || (echo "Ignoring 'docker rmi'" && exit 0)
