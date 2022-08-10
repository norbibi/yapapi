FROM maugnorbert/docker_golem_cuda_base:latest

RUN apt-get update && apt-get install -y \
	python3-pip \
	isc-dhcp-client \
	opensm \
	rdma-core \
	net-tools \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
	&& pip3 install protobuf==3.20.1 pytorch-lightning \
	&& rm -rf /root/.cache/pip

VOLUME /golem/work /golem/input /golem/output /golem/resources
WORKDIR /golem/work

CMD blob
