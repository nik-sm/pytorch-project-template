A simple, extensible project template for using PyTorch.

# Usage

Run `make` to setup environment as needed and format/lint files in the main project module and `scripts` folders.

Run experiments using scripts, e.g.:
```shell
source venv/bin/activate
python scripts/run_experiment.py --name test --epochs 2 --dataset_style map
python scripts/run_experiment.py --name test --epochs 2 --dataset_style iterable
```

# Setup for a new project

<details>
  <summary>Rename <code>pytorch_project_template</code> for your project</summary>

In a python interpreter:
```python
# From the root of repository
from rope.base.project import Project
from rope.refactor.rename import Rename
proj = Project(".")
folder = proj.get_module("pytorch_project_template").get_resource()
change = Rename(proj, folder).get_changes("NEW_PROJECT_NAME_HERE")      # <- Choose new name
print(change.get_description())
# proj.do(change)                                                       # When satisfied with the changes
```
</details>

For each dataset, define a new `DataModule` inheriting from `BaseDataModule`.

In the case where automatic batching is fine, we can use a ["map-style" dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets) as shown in this example: [pytorch_project_template/data/map_style_dataset.py](pytorch_project_template/data/map_style_dataset.py).

When custom batching is desired, we can use an ["iterable-style" dataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) as shown in this example: [pytorch_project_template/data/iterable_dataset.py](pytorch_project_template/data/iterable_dataset.py)

# Contents

```shell
.
├── Makefile
├── pytorch_project_template            # Main project module
│   ├── data                            # Code related to datasets
│   │   ├── base.py                     # Base DataModule class
│   │   ├── __init__.py
│   │   ├── iterable_dataset.py         # Example of iterable-style dataset
│   │   ├── map_style_dataset.py        # Example of map-style dataset
│   │   └── utils.py
│   ├── __init__.py
│   ├── models
│   │   ├── base.py                     # Base class for models
│   │   ├── classifier.py               # Simple classifier example
│   │   └── __init__.py
│   ├── README.md
│   ├── trainer.py                      # Trainer class to handle train, val, and test loops
│   └── utils.py
├── README.md
├── requirements.txt
├── scripts                             # Scripts for running each experiment
│   ├── README.md
│   └── run_experiment.py
└── setup.py
```
