#!/usr/bin/env python3

import pathlib
import asyncio
import sys
from typing import AsyncIterable

from yapapi import Golem, Task, WorkContext
from yapapi.log import enable_default_logger
from yapapi.payload import vm

examples_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(examples_dir))

from utils import (
    build_parser,
    print_env_info,
)

async def worker(context: WorkContext, tasks: AsyncIterable[Task]):
    async for task in tasks:

        script = context.new_script()
        future_result = script.run("/bin/bash", "-c", "nvidia-smi")
        yield script
        task.accept_result(result=await future_result)

async def main(subnet_tag, payment_driver=None, payment_network=None):
    package = await vm.repo(
        image_hash="198e1eb303ca0ce11dd4cb655ddbf43f85a045990c25d6e55001425a",
        min_mem_gib=1,
        min_cpu_threads=1,
        capabilities=["cuda, GA102 [GeForce RTX 3090]"],
    )

    tasks = [Task(data=None)]

    async with Golem(budget=1.0, subnet_tag=subnet_tag) as golem:
        print_env_info(golem)
        async for completed in golem.execute_tasks(worker, tasks, payload=package):
            print(completed.result.stdout)


if __name__ == "__main__":
    enable_default_logger(log_file="template-cuda.log")
    parser = build_parser("Golem Cuda Template App")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    task = loop.create_task(main(
        subnet_tag=args.subnet_tag,
        payment_driver=args.payment_driver,
        payment_network=args.payment_network
    ))
    loop.run_until_complete(task)

# ./template_cuda.py --subnet-tag devnet-beta

