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

error = False
master_index = None
nonzero_nr = None
done_count = 0

class DdpService(Service):
    @staticmethod
    async def get_payload():
        return await vm.repo(
		image_hash="59548268bddc33a22a7f0c2b28dad813c6375d3ec9f874185a0496ae",
            min_mem_gib=4,
            min_cpu_threads=4,
            min_storage_gib=4,
            capabilities=[vm.VM_CAPS_VPN, "cuda, GA102 [GeForce RTX 3090]", "internet_outbound", "ib, vBEZnbRqM/xiSqydxOrrZA5z1+UqAs/SXkGAgHe9vOQ"],
        )

    async def start(self):

        async for script in super().start():
            yield script

        global error
        global master_index
        global nonzero_nr
        global done_count

        app = args.app

        script_dir = pathlib.Path(__file__).resolve().parent
        app_path = str(script_dir / app)
        
        script = self._ctx.new_script()

        try:

            script.upload_file(app_path, "/golem/resources/app.py")
            script.run("/bin/bash", "-c", "modprobe virtio-pci && modprobe virtio-net && dhcpcd")
            
            golem_ip = ipaddress.IPv4Address(self.network_node.ip)
            ib_ip = golem_ip + (1 << 24)
            cmd_set_ib_ip = 'modprobe ib_qib && modprobe ib_ipoib && modprobe ib_umad && sleep 40 && opensm -B && ibportstate --Direct 0 1 enable && ifconfig ib0 ' + str(ib_ip) + ' && sleep 40'
            script.run("/bin/bash", "-c", cmd_set_ib_ip)

            env = 'NCCL_SOCKET_IFNAME=ib0 '
            env = env + 'MASTER_PORT=1234 '

            ib_ips = [(ipaddress.IPv4Address(instance.network_node.ip) + (1 << 24)) for instance in self.cluster.service_runner.instances]
            index = ib_ips.index(ib_ip)

            if master_index is None:
                nr = 0
                master_index = index
                env = env + 'MASTER_ADDR=' + str(ib_ip) + ' '
            else:
                if nonzero_nr is None:
                    nonzero_nr = 1
                elif nonzero_nr < args.nodes:
                    nonzero_nr = nonzero_nr + 1

                nr = nonzero_nr
                env = env + 'MASTER_ADDR=' + str(ib_ips[master_index]) + ' '
          
            env = env + 'NODE_RANK=' + str(nr)
            cmd = 'export ' + env + ' && cd /golem/resources && python3 app.py --nodes ' + str(args.nodes) + ' --gpus ' + str(args.gpus) + ' --batchsize ' + str(args.batchsize) + ' --epochs ' + str(args.epochs) + ' --numworkers ' + str(args.numworkers) + ' || true'
            print(cmd)
            future_result = script.run("/bin/bash", "-c", cmd)
            yield script
            stdresult = await future_result

            print(stdresult.stderr)
            print(stdresult.stdout)

            done_count = done_count + 1

        except:
            error = True


    async def reset(self):
        pass


async def main(subnet_tag, payment_driver=None, payment_network=None, num_instances=1):
    async with Golem(
        budget=1.0,
        subnet_tag=subnet_tag,
        payment_driver=payment_driver,
        payment_network=payment_network,
    ) as golem:

        global error

        print_env_info(golem)

        network = await golem.create_network("192.168.0.1/24")
        async with network:

            cluster = await golem.run_service(DdpService, network=network, num_instances=num_instances)
            instances = cluster.instances

            while error is False:
                try:
                    if done_count == num_instances:
                        break
                    else:
                        await asyncio.sleep(5)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    break

            cluster.stop()


if __name__ == "__main__":
    parser = build_parser("Golem Pytorch")
    parser.add_argument(
        "--nodes",
        type=int,
        default=2,
        help="Number of nodes"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of gpus per node"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--numworkers",
        type=int,
        default=1,
        help="Number of workers per node"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--app",
        type=str,
        required=True,
        help="Python app fullpath"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset fullpath"
    )
    parser.set_defaults(log_file=f"pytorch_cuda.log")
    args = parser.parse_args()

    run_golem_example(
        main(
            subnet_tag=args.subnet_tag,
            payment_driver=args.payment_driver,
            payment_network=args.payment_network,
            num_instances=args.nodes
        ),
        log_file=args.log_file,
    )

# ./pytorch_cuda.py --nodes 2 --gpus 1 --subnet-tag devnet-beta --batchsize 55000 --numworkers 4 --epochs 100 --app dist_train.py
