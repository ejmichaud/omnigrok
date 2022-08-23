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
        f"dnn=False",
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
  init_scale = "exp-logs/{dataset_name}/init-scale_{init_scale}_weight-decay_{weight_decay}_train-points_{train_points}_seed_{seed}"
  learning_rate = "exp-logs/lr_last_layer/lr_last_layer_{lr_last_layer}_init_scale_{init_scale}_weight_decay_{weight_decay}_seed_{seed}"
  learning_rate_cnn = "exp-logs/lr_last_layer_cnn/lr_last_layer_{lr_last_layer}_init_scale_{init_scale}_weight_decay_{weight_decay}_seed_{seed}"

class Configs:
  init_scales = [1,9,21,45]
  seeds = list(range(1,4))
  weight_decays = [0, 1e-4, 1e-3, 1e-2, 1e-1]
  dataset_names = ["CIFAR10"]
  train_points = [1000]

  last_layer_init_scales = [1,3,5,7,9]
  lr_last_layer = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

rule last_layer:
  input:
    expand(Locations.learning_rate_cnn,
            init_scale = Configs.last_layer_init_scales,
            weight_decay = Configs.weight_decays,
            lr_last_layer = Configs.lr_last_layer,
            seed = Configs.seeds,
            )


rule init_scale_test:
  input:
    expand(Locations.init_scale,
            init_scale=Configs.init_scales,
            dataset_name=Configs.dataset_names,
            weight_decay=Configs.weight_decays,
            train_points=Configs.train_points,
            seed=Configs.seeds),

rule lr_last_layer:
  input:
    script = "scripts/train-mnist-mlp-lastlayer.py",
  output:
    logs = directory(Locations.learning_rate_cnn)
  run:
    cmd = " ".join([f"python {input.script} run with",
                   f"dnn=False",
                   f"dataset_name=MNIST",
                   f"optimization_steps=50000",
                   f"weight_decay={wildcards.weight_decay}",
                   f"train_points=1000",
                   f"test_points=1000",
                   f"initialization_scale={wildcards.init_scale}",
                   f"lr_last_layer={wildcards.lr_last_layer}",
                   f"seed={wildcards.seed}",
                   f"-F {output.logs}"])
    run_on_free_gpu(cmd)

rule with_initialization_scale:
  output: 
    logs = directory(Locations.init_scale)
  run: 
    cmd = train_cmd(output.logs,
                    init_scale=wildcards.init_scale,
                    weight_decay=wildcards.weight_decay,
                    dataset_name=wildcards.dataset_name,
                    optimization_steps=100000,
                    train_points=wildcards.train_points,
                    test_points=1000,
                    seed=wildcards.seed)

    run_on_free_gpu(cmd)
