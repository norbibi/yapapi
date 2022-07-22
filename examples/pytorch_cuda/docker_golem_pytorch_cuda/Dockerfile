FROM golem_tmp:latest
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install protobuf==3.20.1 pytorch-lightning
VOLUME /golem/work /golem/input /golem/output /golem/resources
WORKDIR /golem/work
CMD blob
