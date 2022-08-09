FROM docker_golem_cuda_base:latest

RUN apt-get update && apt-get install -y \
	ffmpeg \
	isc-dhcp-client \
	&& rm -rf /var/lib/apt/lists/*

VOLUME /golem/work /golem/input /golem/output /golem/resources
WORKDIR /golem/work

CMD blob