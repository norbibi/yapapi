#!/usr/bin/python3

import asyncio
import pathlib
import random
import sys
import string
import ipaddress
from uuid import uuid4

from yapapi import (
    Golem,
    __version__ as yapapi_version,
)
from yapapi.log import enable_default_logger, log_summary, log_event_repr
from yapapi.payload import vm
from yapapi.services import Service

examples_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(examples_dir))

from utils import (
    build_parser,
    TEXT_COLOR_CYAN,
    TEXT_COLOR_DEFAULT,
    TEXT_COLOR_RED,
    TEXT_COLOR_YELLOW,
    run_golem_example,
    print_env_info,
)

done = False

class FfmpegService(Service):
    @staticmethod
    async def get_payload():
        return await vm.repo(
		image_hash="45f7d8eb72a9360ddffbd9e846ea3617166013ea2a3553eac53cc090",
            min_mem_gib=8,
            min_cpu_threads=8,
            min_storage_gib=8,
            capabilities=[vm.VM_CAPS_VPN, "cuda, GA102 [GeForce RTX 3090]", "internet_outbound"],
        )

    async def start(self):

        async for script in super().start():
            yield script

        global done

        script = self._ctx.new_script()

        if args.url is not None:
            cmd = "wget " + args.url + " -O /golem/input/video_in.mp4"
            script.run("/bin/bash", "-c", cmd)
        elif args.gdriveid is not None:
            script.run("/bin/bash", "-c", "modprobe virtio-pci && modprobe virtio-net && dhclient eth0 && echo 'nameserver 8.8.8.8' > /etc/resolv.conf || true")
            cmd = "wget \"https://docs.google.com/uc?export=download&confirm=$(wget 'https://docs.google.com/uc?export=download&id=" + args.gdriveid + "' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=" +  args.gdriveid + "\" -O /golem/input/video_in.mp4"
            script.run("/bin/bash", "-c", cmd)
        elif args.file is not None:
            file = args.file
            script_dir = pathlib.Path(__file__).resolve().parent
            file_path = str(script_dir / file)
            script.upload_file(file_path, "/golem/input/video_in.mp4")
        else:
            print("Error, Please specify url or gdriveid or file option")
            return

        future_result = script.run("/bin/bash", "-c", "ffmpeg -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 8 -autorotate 0 -i /golem/input/video_in.mp4 -c:a copy -c:v hevc_nvenc -b:v 5M /golem/output/video_out.mp4")
        script.download_file("/golem/output/video_out.mp4", "video_out.mp4")

        yield script
        stdresult = await future_result

        print(stdresult.stderr)
        print(stdresult.stdout)

        done = True


    async def reset(self):
        pass


async def main(subnet_tag, payment_driver=None, payment_network=None, num_instances=1):
    async with Golem(
        budget=1.0,
        subnet_tag=subnet_tag,
        payment_driver=payment_driver,
        payment_network=payment_network,
    ) as golem:

        global done

        print_env_info(golem)

        network = await golem.create_network("192.168.0.1/24")
        async with network:

            cluster = await golem.run_service(FfmpegService, network=network, num_instances=num_instances)
            instances = cluster.instances

            while done is False:
                try:
                    await asyncio.sleep(5)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    break

            cluster.stop()


if __name__ == "__main__":
    parser = build_parser("Golem Cuda FFMPEG")

    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default=None,
        help="Video path url"
    )
    parser.add_argument(
        "--gdriveid",
        type=str,
        required=False,
        default=None,
        help="Google Drive ID"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        default=None,
        help="Local video path"
    )

    parser.set_defaults(log_file=f"cuda_ffmpeg.log")
    args = parser.parse_args()

    if args.url is None and args.gdriveid is None and args.file is None:
        print("Error, Please specify url or gdriveid or file option")
    else:
        run_golem_example(
            main(
                subnet_tag=args.subnet_tag,
                payment_driver=args.payment_driver,
                payment_network=args.payment_network,
            ),
            log_file=args.log_file,
        )

# ./cuda_ffmpeg.py --subnet-tag devnet-beta --gdriveid 1BNzHX9lYP7G4-ensM0o64z4d4asTK2nD
# ./cuda_ffmpeg.py --subnet-tag devnet-beta --file video.mp4
