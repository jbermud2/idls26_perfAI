# Installation

Creating the environment using the .yml:

```conda env create -f environment.yml --prefix /ocean/projects/cis260045p/shared/perf_ai```

Activating the environment:
```conda activate /ocean/projects/cis260045p/shared/perf_all```

Location of MNIST:

```cd /ocean/projects/cis260045p/shared/data/MNIST```

psc flowers data guide

dataset/train — training images
dataset/valid — validation images
dataset/test — test images
cat_to_name.json — maps category numbers to flower names

README.md — dataset info
The dataset is fully downloaded and extracted at:
```/ocean/projects/cis260045p/shared/data/flowers```
load it in your notebook using this path. 
For example in PyTorch:

data_dir = ```'/ocean/projects/cis260045p/shared/data/flowers/dataset'```

train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir  = data_dir + '/test'
