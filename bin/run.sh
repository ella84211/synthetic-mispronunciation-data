#!/bin/bash
set -e


# Activating the environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mdd_env

# Baselines
echo "Running the baselines..."

python baselines/scripts/majority_class.py
python baselines/scripts/random_bias.py

echo "Baseline results are located in baselines/predictions."

# Training
echo "Getting vocabulary..."
python training/get_vocab.py

echo "Training model on only real data..."
python training/train.py train --train data/real_data/train.json --output experiments/experiment_real

echo "Training model on only synthetic data..."
python training/train.py train --train data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json --output experiments/experiment_synthetic

echo "Training model on both real and synthetic data..."
python training/train.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json --output experiments/experiment_both

echo "Finished training!"

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/experiment_real/training_log.txt --output_dir experiments/experiment_real
python training/loss_graph.py experiments/experiment_synthetic/training_log.txt --output_dir experiments/experiment_synthetic
python training/loss_graph.py experiments/experiment_both/training_log.txt --output_dir experiments/experiment_both

# Evaluation
echo "Now evaluating results..."

python evaluation/evaluation.py compare --conditions majority:baselines/predictions/majority_class_predictions.json random:baselines/predictions/random_bias_predictions.json real_only:experiments/experiment_real/predictions.json synthetic_only:experiments/experiment_synthetic/predictions.json both:experiments/experiment_both/predictions.json > scores.txt

echo "Results written to scores.txt"