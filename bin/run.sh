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

for model in bilstm transformer gru
do
    echo "
    ------------------------------"
    echo "Now training ${model}"
    echo "------------------------------
    "

    echo "Training ${model} on only real data..."
    python training/train_${model}.py train --train data/real_data/train.json --output experiments/${model}_real

    echo "Training model on both real and synthetic data..."
    python training/train_${model}.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json --output experiments/${model}_1
    python training/train_${model}.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json --output experiments/${model}_2
    python training/train_${model}.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json --output experiments/${model}_3
    python training/train_${model}.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json --output experiments/${model}_4
    python training/train_${model}.py train --train data/real_data/train.json data/synthetic_data/synthetic_transcriptions_set1.json data/synthetic_data/synthetic_transcriptions_set2.json data/synthetic_data/synthetic_transcriptions_set3.json data/synthetic_data/synthetic_transcriptions_set4.json data/synthetic_data/synthetic_transcriptions_set5.json --output experiments/${model}_5
    echo "Finished training on synthetic data."

    # Get loss graphs
    echo "Getting loss graphs..."
    python training/loss_graph.py experiments/${model}_real/training_log.txt --output_dir experiments/${model}_real
    for ((i=1; i<6; i++))
    do
        python training/loss_graph.py experiments/${model}_$i/training_log.txt --output_dir experiments/${model}_$i
    done

    # Evaluation
    echo "Now getting significance between real data only and synthetic data..."
    for ((i=1; i<6; i++))
    do
        mkdir -p results/${model}_$i
        python evaluation/significance.py --pred_a experiments/${model}_real/predictions.json --pred_b experiments/${model}_$i/predictions.json > results/${model}_$i/significance.txt
        echo "Results written to results/${model}_$i/significance.txt"
    done

    echo "Now getting per-sample scores..."
    mkdir -p results/${model}_real
    python evaluation/per_sample.py --input experiments/${model}_real/predictions.json --output results/${model}_real/per_sample_scores.tsv
    echo "Per-sample scores written to results/${model}_real/per_sample_scores.tsv
    for ((i=1; i<6; i++))
    do
        python evaluation/per_sample.py --input experiments/${model}_$i/predictions.json --output results/${model}_$i/per_sample_scores.tsv
        echo "Per-sample scores written to results/${model}_$i/per_sample_scores.tsv"
    done
done


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
    > results/all_scores.txt
echo "Metrics written to results/all_scores.txt"
