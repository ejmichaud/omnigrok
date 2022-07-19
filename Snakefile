from pytools.persistent_dict import PersistentDict
from time import sleep
import threading

gpu_jobs = PersistentDict("jobs")

onstart:
  gpu_jobs.store("gpu0", 0)
  gpu_jobs.store("gpu1", 0)

lock = threading.Lock()

def run_on_free_gpu(cmd, max_jobs_per_gpu=5):
  while True:
    with lock:
      gc0, gc1 = (gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1"))
      if not (gc0 >= max_jobs_per_gpu and gc1 >= max_jobs_per_gpu):
        cuda_id = 0 if gc0 <= gc1 else 1
        count = gc0 if cuda_id == 0 else gc1
        gpu_jobs.store(f"gpu{cuda_id}", count + 1)
        print(f"running on GPU {cuda_id}")
        shell(cmd + f" device=\"cuda:{cuda_id}\"")
        gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") - 1)
        break
    sleep(15)
    print("waiting...")

def train_cmd(path, init_scale, optimization_steps, seed):
  return " ".join(
    [
        f"python scripts/train-mnist-mlp.py run with",
        f"dnn=False",
        f"initialization_scale={init_scale}",
        f"optimization_steps={optimization_steps}",
        f"seed={seed}",
        f"-F {path}",
    ]
  )
class Locations:
  init_scale = "logs/init-scale_{init_scale}_seed_{seed}"

class Configs:
  init_scales = [1,2,3,4,5,6]
  seeds = list(range(1,6))


rule all:
  input:
    expand(Locations.init_scale, init_scale=Configs.init_scales, seed=Configs.seeds),
    

rule with_initialization_scale:
  output: 
    logs = directory(Locations.init_scale)
  run: 
    cmd = train_cmd(output.logs,
                    init_scale=wildcards.init_scale,
                    optimization_steps=100000,
                    seed=wildcards.seed)
    run_on_free_gpu(cmd)
