FROM docker_golem_cuda_base:latest

RUN apt-get update && apt-get install -y \
	openssh-server \
	&& rm -rf /var/lib/apt/lists/*

CMD blob
