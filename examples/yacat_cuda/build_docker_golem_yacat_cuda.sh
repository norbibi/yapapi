#!/bin/sh

. ../../../build_images_helper.sh

dir=$(dirname "$(realpath "$0")")
build_golem_tmp_fs qemu_x86_64_golem_yacat_cuda_defconfig
build_docker_golem_tmp $dir
build_docker_image docker_golem_yacat_cuda yacat_cuda.py
clean

