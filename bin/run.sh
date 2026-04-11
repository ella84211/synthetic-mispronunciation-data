#!/bin/bash
set -e


# Activating the environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mdd_env

# Make results directory
mkdir -p results/

# Training
echo "Getting vocabulary..."
python training/get_vocab.py

# BiLSTM
echo "
------------------------------"
echo "BiLSTM"
echo "------------------------------
"

echo "Training model on only real data..."
python training/train_bilstm.py train --train data/real_data/train.json --output experiments/bilstm_real

# One set of synthetic data
echo "Training model on both real and synthetic data..."
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json --output experiments/bilstm_1
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json --output experiments/bilstm_2
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json --output experiments/bilstm_3
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json --output experiments/bilstm_4
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json data/synthetic_data/synthetic_transcriptions_set5.json --output experiments/bilstm_5
echo "Finished training on synthetic data."

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/bilstm_real/training_log.txt --output_dir experiments/bilstm_real
python training/loss_graph.py experiments/bilstm_1/training_log.txt --output_dir experiments/bilstm_1
python training/loss_graph.py experiments/bilstm_2/training_log.txt --output_dir experiments/bilstm_2
python training/loss_graph.py experiments/bilstm_3/training_log.txt --output_dir experiments/bilstm_3
python training/loss_graph.py experiments/bilstm_4/training_log.txt --output_dir experiments/bilstm_4
python training/loss_graph.py experiments/bilstm_5/training_log.txt --output_dir experiments/bilstm_5

# Evaluation
echo "Now getting significance between real data only and synthetic data..."
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_1/predictions.json > results/bilstm_1_significance.txt
echo "Results written to results/bilstm_1_significance.txt"
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_2/predictions.json > results/bilstm_2_significance.txt
echo "Results written to results/bilstm_2_significance.txt"
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_3/predictions.json > results/bilstm_3_significance.txt
echo "Results written to results/bilstm_3_significance.txt"
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_4/predictions.json > results/bilstm_4_significance.txt
echo "Results written to results/bilstm_4_significance.txt"
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_5/predictions.json > results/bilstm_5_significance.txt
echo "Results written to results/bilstm_5_significance.txt"


# Transformer
echo "
------------------------------"
echo "Transformer"
echo "------------------------------
"

echo "Training model on only real data..."
python training/train_transformer.py train --train data/real_data/train.json --output experiments/transformer_real

# No oversampling
echo "Training model on both real and synthetic data..."
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json --output experiments/transformer_1
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json --output experiments/transformer_2
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json --output experiments/transformer_3
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json --output experiments/transformer_4
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json data/synthetic_data/synthetic_transcriptions_set5.json --output experiments/transformer_5
echo "Finished training on synthetic data."

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/transformer_real/training_log.txt --output_dir experiments/transformer_real
python training/loss_graph.py experiments/transformer_1/training_log.txt --output_dir experiments/transformer_1
python training/loss_graph.py experiments/transformer_2/training_log.txt --output_dir experiments/transformer_2
python training/loss_graph.py experiments/transformer_3/training_log.txt --output_dir experiments/transformer_3
python training/loss_graph.py experiments/transformer_4/training_log.txt --output_dir experiments/transformer_4
python training/loss_graph.py experiments/transformer_5/training_log.txt --output_dir experiments/transformer_5

# Evaluation
echo "Now getting significance between real data only and synthetic data..."
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_1/predictions.json > results/transformer_1_significance.txt
echo "Results written to results/transformer_1_significance.txt"
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_2/predictions.json > results/transformer_2_significance.txt
echo "Results written to results/transformer_2_significance.txt"
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_3/predictions.json > results/transformer_3_significance.txt
echo "Results written to results/transformer_3_significance.txt"
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_4/predictions.json > results/transformer_4_significance.txt
echo "Results written to results/transformer_4_significance.txt"
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_5/predictions.json > results/transformer_5_significance.txt
echo "Results written to results/transformer_5_significance.txt"


# GRU
echo "
------------------------------"
echo "GRU"
echo "------------------------------
"

echo "Training model on only real data..."
python training/train_gru.py train --train data/real_data/train.json --output experiments/gru_real

# No oversampling
echo "Training model on both real and synthetic data..."
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json --output experiments/gru_1
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json --output experiments/gru_2
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json --output experiments/gru_3
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json --output experiments/gru_4
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json data/synthetic_data/synthetic_transcriptions_set5.json --output experiments/gru_5
echo "Finished training on synthetic data."

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/gru_real/training_log.txt --output_dir experiments/gru_real
python training/loss_graph.py experiments/gru_1/training_log.txt --output_dir experiments/gru_1
python training/loss_graph.py experiments/gru_2/training_log.txt --output_dir experiments/gru_2
python training/loss_graph.py experiments/gru_3/training_log.txt --output_dir experiments/gru_3
python training/loss_graph.py experiments/gru_4/training_log.txt --output_dir experiments/gru_4
python training/loss_graph.py experiments/gru_5/training_log.txt --output_dir experiments/gru_5

# Evaluation
echo "Now getting significance between real data only and non-oversampled synthetic data..."
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_1/predictions.json > results/gru_1_significance.txt
echo "Results written to results/gru_1_significance.txt"
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_2/predictions.json > results/gru_2_significance.txt
echo "Results written to results/gru_2_significance.txt"
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_3/predictions.json > results/gru_3_significance.txt
echo "Results written to results/gru_3_significance.txt"
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_4/predictions.json > results/gru_4_significance.txt
echo "Results written to results/gru_4_significance.txt"
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_5/predictions.json > results/gru_5_significance.txt
echo "Results written to results/gru_5_significance.txt"


# All
echo "Now computing all metrics for all models..."
python evaluation/compute_metrics.py compare --conditions \
    bilstm_r:experiments/bilstm_real/predictions.json bilstm_1:experiments/bilstm_1/predictions.json \
    bilstm_2:experiments/bilstm_2/predictions.json bilstm_3:experiments/bilstm_3/predictions.json \
    bilstm_4:experiments/bilstm_4/predictions.json bilstm_5:experiments/bilstm_5/predictions.json \
    transformer_r:experiments/transformer_real/predictions.json transformer_1:experiments/transformer_1/predictions.json \
    transformer_2:experiments/transformer_2/predictions.json transformer_3:experiments/transformer_3/predictions.json \
    transformer_4:experiments/transformer_4/predictions.json transformer_5:experiments/transformer_5/predictions.json \
    gru_r:experiments/gru_real/predictions.json gru_1:experiments/gru_1/predictions.json \
    gru_2:experiments/gru_2/predictions.json gru_3:experiments/gru_3/predictions.json \
    gru_4:experiments/gru_4/predictions.json gru_5:experiments/gru_5/predictions.json \
    > results/scores.txt
echo "Metrics written to results/scores.txt"
