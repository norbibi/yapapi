#!/usr/bin/env python3

from datetime import datetime, timedelta
import pathlib
import sys

from yapapi import (
    Golem,
    Task,
    WorkContext,
)
from yapapi.payload import vm
from yapapi.rest.activity import BatchTimeoutError

examples_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(examples_dir))

from utils import (
    build_parser,
    TEXT_COLOR_CYAN,
    TEXT_COLOR_DEFAULT,
    TEXT_COLOR_RED,
    TEXT_COLOR_MAGENTA,
    format_usage,
    run_golem_example,
    print_env_info,
)

async def main(subnet_tag, min_cpu_threads, cycles_device, scene, payment_driver=None, payment_network=None, show_usage=False):
    package = await vm.repo(
		image_hash="aee4da1c8da0d05aba6aaa7a7bbdc49a69f5210b7fa2b42900ad7db5",
        min_mem_gib=4,
        min_storage_gib=4,
        min_cpu_threads=4,
        capabilities=[vm.VM_CAPS_VPN, "cuda, GA102 [GeForce RTX 3090]"],
    )

    async def worker(ctx: WorkContext, tasks):
        script_dir = pathlib.Path(__file__).resolve().parent
        scene_path = str(script_dir / scene)
        
        script = ctx.new_script(timeout=timedelta(minutes=10))
        script.upload_file(scene_path, "/golem/resources/scene.blend")

        async for task in tasks:
            frame = task.data
            crops = [{"outfilebasename": "out", "borders_x": [0.0, 1.0], "borders_y": [0.0, 1.0]}]
            script.upload_json(
                {
                    "scene_file": "/golem/resources/scene.blend",
                    "resolution": (400, 300),
                    "use_compositing": False,
                    "crops": crops,
                    "samples": 100,
                    "frames": [frame],
                    "output_format": "PNG",
                    "cycles_device": cycles_device,
                    "RESOURCES_DIR": "/golem/resources",
                    "WORK_DIR": "/golem/work",
                    "OUTPUT_DIR": "/golem/output",
                },
                "/golem/work/params.json",
            )

            script.run("/golem/entrypoints/run-blender.sh")
            output_file = f"output_{frame}.png"
            script.download_file(f"/golem/output/out{frame:04d}.png", output_file)
            try:
                yield script
                # TODO: Check if job results are valid
                # and reject by: task.reject_task(reason = 'invalid file')
                task.accept_result(result=output_file)
            except BatchTimeoutError:
                print(
                    f"{TEXT_COLOR_RED}"
                    f"Task {task} timed out on {ctx.provider_name}, time: {task.running_time}"
                    f"{TEXT_COLOR_DEFAULT}"
                )
                raise
            
            script = ctx.new_script(timeout=timedelta(minutes=10))

            if show_usage:
                raw_state = await ctx.get_raw_state()
                usage = format_usage(await ctx.get_usage())
                cost = await ctx.get_cost()
                print(
                    f"{TEXT_COLOR_MAGENTA}"
                    f" --- {ctx.provider_name} STATE: {raw_state}\n"
                    f" --- {ctx.provider_name} USAGE: {usage}\n"
                    f" --- {ctx.provider_name}  COST: {cost}"
                    f"{TEXT_COLOR_DEFAULT}"
                )

    frames: range = range(0, 60, 10)
    init_overhead = 3
    min_timeout, max_timeout = 6, 30

    timeout = timedelta(minutes=max(min(init_overhead + len(frames) * 2, max_timeout), min_timeout))

    async with Golem(
        budget=1.0,
        subnet_tag=subnet_tag,
        payment_driver=payment_driver,
        payment_network=payment_network,
    ) as golem:
        print_env_info(golem)

        num_tasks = 0
        start_time = datetime.now()

        completed_tasks = golem.execute_tasks(
            worker,
            [Task(data=frame) for frame in frames],
            payload=package,
            max_workers=3,
            timeout=timeout,
        )
        async for task in completed_tasks:
            num_tasks += 1
            print(
                f"{TEXT_COLOR_CYAN}"
                f"Task computed: {task}, result: {task.result}, time: {task.running_time}"
                f"{TEXT_COLOR_DEFAULT}"
            )

        print(
            f"{TEXT_COLOR_CYAN}"
            f"{num_tasks} tasks computed, total time: {datetime.now() - start_time}"
            f"{TEXT_COLOR_DEFAULT}"
        )


if __name__ == "__main__":
    parser = build_parser("Render a Blender scene")
    parser.add_argument("--show-usage", action="store_true", help="show activity usage and cost")
    
    parser.add_argument(
        "--min-cpu-threads",
        type=int,
        default=2,
        help="require the provider nodes to have at least this number of available CPU threads",
        required=False
    )
    parser.add_argument(
        "--cycles-device",
        type=str,
        default="CUDA",
        help="None, CUDA or CUDA+CPU",
        required=False
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Blend file scene",
        required=True,
    )

    parser.set_defaults(log_file=f"blender-cuda.log")
    args = parser.parse_args()

    run_golem_example(
        main(
            subnet_tag=args.subnet_tag,
            min_cpu_threads=args.min_cpu_threads,
            cycles_device=args.cycles_device,
            scene=args.scene,
            payment_driver=args.payment_driver,
            payment_network=args.payment_network,
            show_usage=args.show_usage
        ),
        log_file=args.log_file,
    )

# ./blender_cuda.py --subnet-tag devnet-beta --scene Blender_3.blend