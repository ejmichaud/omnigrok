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

## Accessing experimental data

I've been running experiments with the `scripts/train-mnist-mlp.py` script on the MIT OpenMind cluster. The results are being saved to a MongoDB. This is a cool setup since it makes the results accessible from any computer, not just from the OpenMind cluster. 

For an example of how to download and visualize this data, see the `notebooks/visualize-mnist.ipynb` notebook. You'll see that this requires SEML to be installed. To install my version of SEML, use: `pip install git+https://github.com/ejmichaud/seml.git`. Then edit `~/.config/seml/mongodb.config` to point to the right MongoDB (I'll provide credentials privately). You should then be able to run the notebook. 

Experiments are organized into MongoDB collections. The directory `seml-configs` contains the experiment names and the configs which specified the parameter search for each experiment.

