FROM docker_golem_cuda_base:latest

RUN apt-get update && apt-get install -y \
	python3-pip \
	libx11-6 \
	libxi6 \
	libxxf86vm1 \
	libxrender1 \
	libxfixes3 \
	&& rm -rf /var/lib/apt/lists/*

ADD https://download.blender.org/release/Blender3.1/blender-3.1.2-linux-x64.tar.xz /opt/
RUN tar -xf /opt/blender-3.1.2-linux-x64.tar.xz -C /opt
RUN mv /opt/blender-3.1.2-linux-x64 /opt/blender
RUN rm /opt/blender-3.1.2-linux-x64.tar.xz
RUN ln -s /opt/blender/blender /usr/bin/blender
COPY entrypoints /golem/entrypoints

VOLUME /golem/work /golem/input /golem/output /golem/resources
WORKDIR /golem/work

CMD blob
