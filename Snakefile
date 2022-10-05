from pytools.persistent_dict import PersistentDict
from time import sleep

onstart:
  gpu_jobs = PersistentDict("jobs")
  gpu_jobs.store("gpu0", 0)
  gpu_jobs.store("gpu1", 0)
  print("gpu_jobs:", (gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1")))
  sleep(.5)

def run_on_free_gpu(cmd, max_jobs_per_gpu=5):
  gpu_jobs = PersistentDict("jobs")
  while True:
    gc0, gc1 = (gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1"))
    if not (gc0 >= max_jobs_per_gpu and gc1 >= max_jobs_per_gpu):
      print("testitest")
      cuda_id = 0 if gc0 <= gc1 else 1
      gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") + 1)
      print(f"running on GPU {cuda_id}")
      shell(cmd + f" device=\"cuda:{cuda_id}\"")
      gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") - 1)
      break
    sleep(15)
    print("waiting...")

def train_cmd(path, init_scale, weight_decay, dataset_name, optimization_steps, train_points, test_points, seed):
  return " ".join(
    [
        f"python scripts/train-mnist-mlp.py run with",
        f"dnn=True",
        f"train_points={train_points}",
        f"test_points={test_points}",
        f"dataset_name={dataset_name}",
        f"initialization_scale={init_scale}",
        f"weight_decay={weight_decay}",
        f"optimization_steps={optimization_steps}",
        f"seed={seed}",
        f"-F {path}",
    ]
  )

class Locations:
  init_scale = "exp-logs/{dataset_name}/dnn_init-scale_{init_scale}_weight-decay_{weight_decay}_train-points_{train_points}_seed_{seed}"

class Configs:
  init_scales = [1,5,9]
  seeds = list(range(1,3))
  weight_decays = [0, 1e-4, 1e-3, 1e-2, 1e-1]
  dataset_names = ["CIFAR10"]
  train_points = [1000]


rule init_scale_test:
  input:
    expand(Locations.init_scale,
            init_scale=Configs.init_scales,
            dataset_name=Configs.dataset_names,
            weight_decay=Configs.weight_decays,
            train_points=Configs.train_points,
            seed=Configs.seeds),


rule with_initialization_scale:
  output: 
    logs = directory(Locations.init_scale)
  run: 
    cmd = train_cmd(output.logs,
                    init_scale=wildcards.init_scale,
                    weight_decay=wildcards.weight_decay,
                    dataset_name=wildcards.dataset_name,
                    optimization_steps=500000,
                    train_points=wildcards.train_points,
                    test_points=1000,
                    seed=wildcards.seed)

    run_on_free_gpu(cmd)
