mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
project_path := $(patsubst %/,%,$(dir $(mkfile_path)))

run: .setup
	docker run -v $(project_path)/:/brkga/ --rm --gpus all brkga

.setup: experiments/Dockerfile experiments/requirements.txt
	docker build --no-cache -t brkga -f experiments/Dockerfile .
	echo "Setup on: $$(date)" >.setup

.PHONY: clean
clean:
	rm -f .setup core
	rm -rf build-*
