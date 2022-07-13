# omnigrok
Grokking all the things

## `scripts`
The script `scripts/train-mnist-mlp.py` does a training run of an MNIST dense neural network, with lots of configuraiton options. To run it with default configuration options and reproduce Ziming's result where generalization improves substantially during overfitting (initialization scale of 7 and weight decay of 0.1), run like so:
```
python scripts/train-mnist-mlp.py run
```
This script is built around the `sacred` library for experiment configuration and logging, and also has support for `seml`, which can be used to define grid-searches over configurations and automatically create slurm job arrays to execute them in parallel. One can see all configuration options with:
```
python scripts/train-mnist-mlp.py print_config
INFO - train-mnist-mlp - Running command 'print_config'
INFO - train-mnist-mlp - Started
Configuration (modified, added, typechanged, doc):
  activation = 'ReLU'                # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'
  batch_size = 200
  db_collection = None
  depth = 3                          # the number of nn.Linear modules in the model
  device = device(type='cuda', index=0)    # computing parameters
  download_directory = '/om/user/ericjm/Downloads/'
  dtype = torch.float32
  initialization_scale = 7.0
  log_freq = 200                     # logging parameters
  loss_function = 'MSE'              # 'MSE' or 'CrossEntropy'
  lr = 0.001
  optimization_steps = 100000
  optimizer = 'AdamW'
  overwrite = None
  seed = 473837534                   # the random seed for this experiment
  train_points = 1000                # training parameters
  verbose = True
  weight_decay = 0.1
  width = 200
INFO - train-mnist-mlp - Completed after 0:00:00
```

By default, this script will not save any results. However, one can save results locally to directory `results`, for example, with:
```
python scripts/train-mnist-mlp.py -F results run with optimization_steps=100 log_freq=10
```
This script takes about 10 seconds to run on my machine. 

Note that using SEML for doing searches over configurations takes a bit of setup. I have a custom version of the package which you can install with
```
pip install git+https://github.com/ejmichaud/seml.git
```

