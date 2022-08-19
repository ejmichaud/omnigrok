import pathlib
import os

import json
import requests

from tqdm.auto import tqdm


MODEL_SHORTHAND = "arwen"
PATH = "/data/ericjm/biggrok"

# load mistral_models.json
with open('mistral/mistral_models.json') as f:
    models = json.load(f)

# print(models.keys())

url = models[MODEL_SHORTHAND]["url"]
model_name = url.split('/')[-1]
# print(model_name)
checkpoints = models["arwen"]["checkpoints"]
# print(url)
# print(checkpoints[0])

# os.mkdir(os.path.join(PATH, model_name))
pathlib.Path(os.path.join(PATH, model_name)).mkdir(parents=True, exist_ok=True)
# print(os.path.join(PATH, model_name))

for c in tqdm(checkpoints):
    c_path = os.path.join(PATH, model_name, c)
    pathlib.Path(c_path).mkdir(exist_ok=True)
    # os.mkdir(c_path)

    # download config.json
    config_url = f"{url}/raw/{c}/config.json"
    os.system(f"curl {config_url} > {os.path.join(c_path, 'config.json')}")

    # download vocab.json
    vocab_url = f"{url}/raw/{c}/vocab.json"
    os.system(f"curl {vocab_url} > {os.path.join(c_path, 'vocab.json')}") 

    # download tokenizer_config.json
    tokenizer_config_url = f"{url}/raw/{c}/tokenizer_config.json"
    os.system(f"curl {tokenizer_config_url} > {os.path.join(c_path, 'tokenizer_config.json')}") 

    # download pytorch_model.bin
    pointer_contents = requests.get(f"{url}/raw/{c}/pytorch_model.bin")
    sha256 = pointer_contents.text.split('\n')[1].split(':')[1]
    model_url = f"https://cdn-lfs.huggingface.co/stanford-crfm/{model_name}/{sha256}"
    os.system(f"curl {model_url} > {os.path.join(c_path, 'pytorch_model.bin')}")

