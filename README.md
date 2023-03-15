# dsc180a_capstone_project

This program takes care of all model data loading preprocessing, model building and training, and visualizations regarding our DSC 180 Capstone project.

## Usage

All functionality is used by running `python3 run.py` followed by some arguments in the root directory. The first argument is `exec_type` which indicates what action you would like to take. Possible values are:
```
data     # for data loading

train    # for model training

OT       # for optimal transport

eval     # for gathering results

viz      # for creating visualizations

all      # for doing all of the above

test     # for testing all of the above
         # funcitonality on dummy data
         # (except downloading the data)
```

The second and final argument is `-c` or `--clean`, which, when included cleans the save directories used by the specified `exec_type`.

### Data Loading

The dataset is downloaded from the WILDS project using their python package, which is a prerequisite. To install run:
```
pip install wilds
```

Running the following command in the root project directory will download the dataset to the proper location:
```
python3 run.py data
```

### Model Training

The model we used for this project was a custom CNN built in pytorch and trained entirely on either urban or rural data from one country. To train a model with settings specified in the `config/train.json`, run:
```
python3 run.py train
```
To change which country the model should be trained on, whether an urban/rural model should be trained, the percentile cutoffs the model uses for classification, or other training parameters, please consult `config/train.json`.

To remove all trained models, run:
```
python3 run.py train --clean
```
or
```
python3 run.py train -c
```

### Optimal Transport

Optimal transport was achieved using the python optimal transport package, which is a prerequisite. To install run:
```
pip install ot
```

In our implementation, optimal transport is used for domain adaptation. The goal is to adapt the color profiles of one country to another, in the hope that the CNN trained on only one country can more accurately classify images from another country.
In order to fit the optimal transport to transport images from a source country to a target country, first edit the source and target country fields in `config/OT.json`, then run:
```
python3 run.py OT
```

To remove all saved OT models, run:
```
python3 run.py OT --clean
```
or
```
python3 run.py OT -c
```

### Results

The results are gathered and displayed using data, models and OT objects saved from above. This function will error if run without running `data`, `train`, and `OT` first with aligning configuration to generate objects. To run:
```
python3 run.py eval
```

To remove all saved results, run:
```
python3 run.py eval -c # or --clean
```
or
```
python3 run.py eval -c
```
### Visulisations