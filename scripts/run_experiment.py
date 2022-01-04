from datetime import datetime
from tqdm import tqdm

from loguru import logger
from pytorch_project_template import Classifier, MapStyleMNIST, IterableStyleMNIST, Trainer
from pytorch_project_template.utils import get_git_hash, get_project_paths, seed_everything
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="phrase to remember experiment (UsePascalCase")
p.add_argument("--tiny", action="store_true", default=False, help="1 batch per epoch")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--dataset_style", choices=["map", "iterable"], default="map")
args = p.parse_args()

if args.dataset_style == "map":
    datamodule_class = MapStyleMNIST
else:
    datamodule_class = IterableStyleMNIST

project_path, _, _ = get_project_paths()
base_results = project_path / "results" / args.name
base_results.mkdir(exist_ok=True, parents=True)

logger.warning("NOTE - be sure to re-initialize any trained components between settings!")
all_results = []
for setting in tqdm([1, 2, 3], desc="Experimental Settings"):
    seed_everything(setting)

    now = datetime.now().isoformat("_", "seconds")
    gh = get_git_hash()
    run_name = f"{now}__{gh}__{setting}"

    trainer = Trainer(
        model=Classifier(),
        datamodule=datamodule_class(),
        results_dir=base_results / run_name,
        experimental_setting=setting,
        tqdm_pos=1,
    )
    results = trainer(args.epochs, tiny=args.tiny)
    all_results.append(results)

    # re-create and save every time, in case of interruption
    df = pd.DataFrame.from_records(all_results)
    df.to_pickle(base_results / "all_results.pkl")

print(df)
print(df.describe())
