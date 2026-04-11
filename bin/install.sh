#!/bin/bash
set -e


# Creating and activating the environment
echo "Creating Conda environment..."

source "$(conda info --base)/etc/profile.d/conda.sh"
if grep -w "mdd_env" <<< `conda info --envs`
then
    echo "Conda environment 'mdd_env' already exists."
else
    conda create -n mdd_env python=3.11 -y
fi
conda activate mdd_env
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed successfully in 'mdd_env'."
echo "Environment created! Now getting the data."


# Getting and preprocessing the speechocean762 data
echo "Getting the speechocean762 data..."

git clone https://github.com/jimbozhang/speechocean762.git
mkdir -p data/real_data
cp speechocean762/resource/scores.json data/real_data

echo "Now preprocessing the speechocean762 data..."

python data_preprocessing/real_data/preprocess_speechocean762.py
python data_preprocessing/real_data/split_speechocean762.py

echo "Preprocessed the speechocean data. Located in data/real_data."


# Getting and preprocessing the synthetic data
echo "Getting the base sentences..."

python data_preprocessing/synthetic_data/get_sentences.py
python data_preprocessing/synthetic_data/filter_sentences.py

echo "Making the IPA transcriptions and injecting errors..."

python data_preprocessing/synthetic_data/make_ipa_transcriptions.py
python data_preprocessing/synthetic_data/make_pronunciation_errors.py data/synthetic_data/

echo "Preprocessed the synthetic data. Located in data/synthetic_data."

echo "Installation complete."