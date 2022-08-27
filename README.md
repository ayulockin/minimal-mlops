# Get more out of your ML workflow:
## Minimal MLOps everyone should know
This repository is for the GDG Cloud Kolkata workshop on 27th August, 2022. Learn about minimal MLOps required for any ML practitioner. Find the curated list of resources [here](https://wandb.me/ccd2022).

:star: **Slide Deck**: http://wandb.me/minimal-mlops-deck

## Prerequisite setup

### Free W&B Account
If you don't have a free W&B account follow the steps:
- Visit [wandb.ai/site](https://wandb.ai/site).
- Click on sign-up and follow the signup process. 
- Login

### Authenticate your machine with W&B authorization key
To authenticate any machine to start logging experiments to your W&B account,
- visit [wandb.ai/authorize](https://wandb.ai/authorize) to get the key.
- alternatively, you can visit your settings page to get the key.

## Quickstart

### Install repository

```
1. git clone https://github.com/ayulockin/minimal-mlops
2. cd minimal-mlops
3. pip install -e .
```

### Install the dependencies

`1. pip install -r requirements.txt`

### Configuration

Everything is stitched together using configs file. You can find the config for this repo in `configs/` dir.

### Train without W&B

`python train.py --config configs/baseline.py`

### Train with W&B

`python train.py --config configs/baseline.py --wandb`

### Checkpoint without W&B

`python train.py --config configs/baseline.py --log_model`

### Checkpoint with W&B

`python train.py --config configs/baseline.py --log_model --wandb`

### Log evaluation

`python train.py --config configs/baseline.py --wandb --log_eval`

### Log data

`python log_data.py --config configs/baseline.py`

I hope you find it useful. If you encounter any issue please raise an issue. :)



