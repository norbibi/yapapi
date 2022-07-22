#!/usr/bin/env python3

import asyncio
from datetime import timedelta
import pathlib
import random
import sys
import string
from uuid import uuid4
from yapapi import (
    Golem,
    __version__ as yapapi_version,
)
from yapapi.log import enable_default_logger, log_summary, log_event_repr  # noqa
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


class SshService(Service):
    @staticmethod
    async def get_payload():
        return await vm.repo(
		image_hash="ece0cf8e7815f079b7886187c06136cbb84c744dfc736d88bfab4121",
            min_mem_gib=8,
            min_storage_gib=8,
            capabilities=[vm.VM_CAPS_VPN, "cuda, GA102 [GeForce RTX 3090]"],
        )

    async def start(self):
        async for script in super().start():
            yield script

        password = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))

        script = self._ctx.new_script()

        script.run("/bin/bash", "-c", "ssh-keygen -A")
        script.run("/bin/bash", "-c", f'echo -e "{password}\n{password}" | passwd')
        script.run("/bin/bash", "-c", 'echo "PermitRootLogin yes" | tee -a /etc/ssh/sshd_config')
        script.run("/bin/bash", "-c", "chown root:root /var/empty && /usr/sbin/sshd")
        yield script

        connection_uri = self.network_node.get_websocket_uri(22)
        app_key = self.cluster.service_runner._job.engine._api_config.app_key

        print(
            "Connect with:\n"
            f"{TEXT_COLOR_CYAN}"
            f"ssh -o ProxyCommand='websocat asyncstdio: {connection_uri} --binary -H=Authorization:\"Bearer {app_key}\"' root@{uuid4().hex}"
            f"{TEXT_COLOR_DEFAULT}"
        )

        print(f"{TEXT_COLOR_RED}password: {password}{TEXT_COLOR_DEFAULT}")

    async def reset(self):
        pass


async def main(subnet_tag, payment_driver=None, payment_network=None, num_instances=2):
    async with Golem(
        budget=1.0,
        subnet_tag=subnet_tag,
        payment_driver=payment_driver,
        payment_network=payment_network,
    ) as golem:
        print_env_info(golem)

        network = await golem.create_network("192.168.0.1/24")
        async with network:
            cluster = await golem.run_service(
                SshService, network=network, num_instances=num_instances
            )
            instances = cluster.instances

            while True:
                print(instances)
                try:
                    await asyncio.sleep(5)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    break

            cluster.stop()

            cnt = 0
            while cnt < 3 and any(s.is_available for s in instances):
                print(instances)
                await asyncio.sleep(5)
                cnt += 1


if __name__ == "__main__":
    parser = build_parser("Golem Cuda VPN SSH")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of instances to spawn",
    )

    parser.set_defaults(log_file=f"ssh-cuda.log")
    args = parser.parse_args()

    run_golem_example(
        main(
            subnet_tag=args.subnet_tag,
            payment_driver=args.payment_driver,
            payment_network=args.payment_network,
            num_instances=args.num_instances,
        ),
        log_file=args.log_file,
    )

# ./ssh_cuda.py --subnet-tag devnet-beta

