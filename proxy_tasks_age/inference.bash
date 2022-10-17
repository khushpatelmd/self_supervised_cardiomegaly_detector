#!/bin/sh
echo "Running the inference. It will produce finetuning by freezing initial layers. Final conv block (3 blocks of conv-bn-relu and linear layer unfreezed)."
echo "It will run for 500 epochs and take few hours"
echo "Results will be saved in experiments.csv"
python finetuning.py