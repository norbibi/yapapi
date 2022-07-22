FROM golem_tmp:latest
ADD https://download.blender.org/release/Blender3.1/blender-3.1.2-linux-x64.tar.xz /opt/
RUN tar -xf /opt/blender-3.1.2-linux-x64.tar.xz -C /opt
RUN mv /opt/blender-3.1.2-linux-x64 /opt/blender
RUN rm /opt/blender-3.1.2-linux-x64.tar.xz
RUN ln -s /opt/blender/blender /usr/bin/blender
COPY entrypoints /golem/entrypoints
VOLUME /golem/work /golem/input /golem/output /golem/resources
WORKDIR /golem/work
CMD blob
