mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
project_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)
device := 0

run: .setup
	docker run --env DEVICE=$(device) -v $(project_path)/:/brkga/ -u "id -u $$USER" --rm --gpus device=$(device) brkga

.setup: experiments/Dockerfile experiments/requirements.txt
	docker build --no-cache --build-arg CUDA_VERSION=$(cuda_version) -t brkga -f experiments/Dockerfile .
	echo "Setup on: $$(date)" >.setup

.PHONY: open-terminal
open-terminal:
	docker run -v $(project_path)/:/brkga/ --rm -it ubuntu

.PHONY: fix-git
fix-git: # Rule created due the errors on dl-1
	find .git/objects/ -size 0 -exec rm -f {} \;
	git fetch origin

.PHONY: clean
clean:
	docker run -v $(project_path)/:/brkga/ --rm ubuntu /bin/bash -c 'cd brkga; rm -f .setup; rm -rf build-*'
	docker rmi $$(docker images 'brkga' -a -q) || exit 0
