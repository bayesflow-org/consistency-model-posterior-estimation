## Bayesian Denoising
This folder contains training and evaluation code for the experiment Bayesian Denoising.

The training code is located in `train.py`, and the evaluation code in `evaluate.ipynb`. `merge_results.ipynb` combines the evaluation results into one table.

## Training
For training, we call the training script with the following parameters:
```bash
$ python train.py --initial-learning-rate 5e-4 --tmax 200.0 --batch-size 32 --num-steps 20000 --num-training 2000 --lr-adapt=cosine --method=cmpe --dense-dim 2048 --architecture=naive
$ python train.py --initial-learning-rate 5e-4 --tmax 200.0 --batch-size 256 --num-steps 20000 --num-training 60000 --lr-adapt=cosine --method=cmpe --dense-dim 2048 --architecture=naive
$ python train.py --initial-learning-rate 5e-4 --tmax 200.0 --batch-size 32 --num-steps 20000 --num-training 2000 --lr-adapt=cosine --method=cmpe --architecture=unet
$ python train.py --initial-learning-rate 5e-4 --tmax 200.0 --batch-size 256 --num-steps 20000 --num-training 60000 --lr-adapt=cosine --method=cmpe --architecture=unet
$ python train.py --initial-learning-rate 5e-4 --batch-size 32 --num-steps 20000 --num-training 2000 --lr-adapt=cosine --method=fmpe --dense-dim 2048 --architecture=naive
$ python train.py --initial-learning-rate 5e-4 --batch-size 256 --num-steps 20000 --num-training 60000 --lr-adapt=cosine --method=fmpe --dense-dim 2048 --architecture=naive
$ python train.py --initial-learning-rate 5e-4 --batch-size 32 --num-steps 20000 --num-training 2000 --lr-adapt=cosine --method=fmpe --architecture=unet
$ python train.py --initial-learning-rate 5e-4 --batch-size 256 --num-steps 20000 --num-training 60000 --lr-adapt=cosine --method=fmpe --architecture=unet
```

## Evaluation
By running `evaluation.ipynb`, all figures and metrics are calculated. Please note that sampling is quite slow for FMPE, so this will take a couple of hours.
