#!/bin/bash
set -e


# Activating the environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mdd_env

# Training
echo "Getting vocabulary..."
python training/get_vocab.py


# BiLSTM
echo "\\n------------------------------"
echo "BiLSTM"
echo "------------------------------\\n"

echo "Training model on only real data..."
python training/train_bilstm.py train --train data/real_data/train.json --output experiments/bilstm_real

# No oversampling
echo "Training model on both real and synthetic data..."
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/not_oversampled/synthetic_transcriptions_set1.json --output experiments/bilstm_1
echo "Finished training with no oversampling. Now training with oversampling factor 1.5..."

# Oversampling factor 1.5
echo "Training model on both real and synthetic data..."
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/oversampled_15/synthetic_transcriptions_set1.json --output experiments/bilstm_15
echo "Finished training with oversampling factor 1.5. Now training with oversampling factor 2..."

# Oversample factor 2
echo "Training model on both real and synthetic data..."
python training/train_bilstm.py train --train data/real_data/train.json data/synthetic_data/oversampled_2/synthetic_transcriptions_set1.json --output experiments/bilstm_2
echo "Finished training BiLSTM!"

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/bilstm_real/training_log.txt --output_dir experiments/bilstm_real
python training/loss_graph.py experiments/bilstm_1/training_log.txt --output_dir experiments/bilstm_1
python training/loss_graph.py experiments/bilstm_15/training_log.txt --output_dir experiments/bilstm_15
python training/loss_graph.py experiments/bilstm_2/training_log.txt --output_dir experiments/bilstm_2

# Evaluation
echo "Now getting significance between real data only and non-oversampled synthetic data..."
python evaluation/significance.py --pred_a experiments/bilstm_real/predictions.json --pred_b experiments/bilstm_1/predictions.json > bilstm_significance.txt
echo "Results written to bilstm_significance.txt"


# Transforer
echo "\\n------------------------------"
echo "Transformer"
echo "------------------------------\\n"

echo "Training model on only real data..."
python training/train_transformer.py train --train data/real_data/train.json --output experiments/transformer_real

# No oversampling
echo "Training model on both real and synthetic data..."
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/not_oversampled/synthetic_transcriptions_set1.json --output experiments/transformer_1
echo "Finished training with no oversampling. Now training with oversampling factor 1.5..."

# Oversampling factor 1.5
echo "Training model on both real and synthetic data..."
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/oversampled_15/synthetic_transcriptions_set1.json --output experiments/transformer_15
echo "Finished training with oversampling factor 1.5. Now training with oversampling factor 2..."

# Oversample factor 2
echo "Training model on both real and synthetic data..."
python training/train_transformer.py train --train data/real_data/train.json data/synthetic_data/oversampled_2/synthetic_transcriptions_set1.json --output experiments/transformer_2
echo "Finished training Transformer!"

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/transformer_real/training_log.txt --output_dir experiments/transformer_real
python training/loss_graph.py experiments/transformer_1/training_log.txt --output_dir experiments/transformer_1
python training/loss_graph.py experiments/transformer_15/training_log.txt --output_dir experiments/transformer_15
python training/loss_graph.py experiments/transformer_2/training_log.txt --output_dir experiments/transformer_2

# Evaluation
echo "Now getting significance between real data only and non-oversampled synthetic data..."
python evaluation/significance.py --pred_a experiments/transformer_real/predictions.json --pred_b experiments/transformer_1/predictions.json > transformer_significance.txt
echo "Results written to transformer_significance.txt"


# GRU
echo "\\n------------------------------"
echo "GRU"
echo "------------------------------\\n"

echo "Training model on only real data..."
python training/train_gru.py train --train data/real_data/train.json --output experiments/gru_real

# No oversampling
echo "Training model on both real and synthetic data..."
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/not_oversampled/synthetic_transcriptions_set1.json --output experiments/gru_1
echo "Finished training with no oversampling. Now training with oversampling factor 1.5..."

# Oversampling factor 1.5
echo "Training model on both real and synthetic data..."
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/oversampled_15/synthetic_transcriptions_set1.json --output experiments/gru_15
echo "Finished training with oversampling factor 1.5. Now training with oversampling factor 2..."

# Oversample factor 2
echo "Training model on both real and synthetic data..."
python training/train_gru.py train --train data/real_data/train.json data/synthetic_data/oversampled_2/synthetic_transcriptions_set1.json --output experiments/gru_2
echo "Finished training GRU!"

# Get loss graphs
echo "Getting loss graphs..."
python training/loss_graph.py experiments/gru_real/training_log.txt --output_dir experiments/gru_real
python training/loss_graph.py experiments/gru_1/training_log.txt --output_dir experiments/gru_1
python training/loss_graph.py experiments/gru_15/training_log.txt --output_dir experiments/gru_15
python training/loss_graph.py experiments/gru_2/training_log.txt --output_dir experiments/gru_2

# Evaluation
echo "Now getting significance between real data only and non-oversampled synthetic data..."
python evaluation/significance.py --pred_a experiments/gru_real/predictions.json --pred_b experiments/gru_1/predictions.json > gru_significance.txt
echo "Results written to gru_significance.txt"


# All
echo "Now computing all metrics for all models..."
python evaluation/compute_metrics.py compare --conditions \
    bilstm_r:experiments/bilstm_real/predictions.json bilstm_1:experiments/bilstm_1/predictions.json \
    bilstm_15:experiments/bilstm_15/predictions.json bilstm_2:experiments/bilstm_2/predictions.json \
    transformer_r:experiments/transformer_real/predictions.json transformer_1:experiments/transformer_1/predictions.json \
    transformer_15:experiments/transformer_15/predictions.json transformer_2:experiments/transformer_2/predictions.json \
    gru_r:experiments/gru_real/predictions.json gru_1:experiments/gru_1/predictions.json \
    gru_15:experiments/gru_15/predictions.json gru_2:experiments/gru_2/predictions.json \
    > scores.txt
echo "Metrics written to scores.txt"