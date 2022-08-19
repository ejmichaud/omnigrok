import json
import os

from tqdm.auto import tqdm

# load mistral_models.json
with open('mistral/mistral_models.json') as f:
    models = json.load(f)

print(models.keys())

url = models["arwen"]["url"]
checkpoints = models["arwen"]["checkpoints"]
print(url)
print(checkpoints[0])

# for c in tqdm(checkpoints):
#   os.system(f"git clone {url} --branch {c} --single-branch")
#   dirname = url.split('/')[-1]
#   os.system(f"cd {dirname}")
#   os.system("git lfs pull")
#   os.system("rm -rf .git")
#   os.system("cd ../")
#   os.system(f"mv {dirname} {dirname}-{c}")


