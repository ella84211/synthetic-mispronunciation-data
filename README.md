# Synthetic Data for Mispronunciation Detection

This repository is for a research project exploring the effect of synthetically augmented data for a mispronunciation detection task. The model, a BiLSTM, will be trained on speech transcriptions in the International Phonetic Alphabet (IPA) that contain synthetic errors. This repository contains all code for data preprocessing, training the models, and evaluating the results.

## Directory Structure

```
synthetic-mispronunciation-data/
├── baselines/
│   └── predictions/
│       ├── majority_class_results.json
│       └── random_bias_80_results.json
│   └── scripts/
│       ├── majority_class.py
│       └── random_bias.py
├── bin/
│   └── install.sh
├── data/
│   └── real_data/
│       ├── scores.json
│       ├── test.json
│       ├── train.json
│       ├── transcriptions.json
│       └── validate.json
│   └── synthetic_data/
│       ├── filtered_sentences.json
│       ├── sentences.json
│       ├── synthetic_transcriptions_set*.json
│       ├── transcriptions.json
│       └── transcriptions_set*.json
├── data_preprocessing/
│   └── real_data/
│       ├── preprocess_speechocean762.py
│       └── split_speechocean762.py
│   └── synthetic_data/
│       ├── filter_sentences.py
│       ├── get_sentences.py
│       ├── make_ipa_transcriptions.py
│       └── make_pronunciation_errors.py
│   └── format_data.py
├── evaluation/
│   └── evaluation.py
├── training/
│   ├── get_vocab.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Clone the repo
```
git clone https://github.com/ella84211/synthetic-mispronunciation-data.git
cd synthetic-mispronunciation-data
```

Create environment and install dependencies (Linux)
```
./bin/install.sh
```

If permission denied error:
```
chmod +x bin/install.sh
./bin/install.sh
```

## Running the code
```
./bin/run.sh
```

If permission denied error:
```
chmod +x bin/run.sh
./bin/run.sh
```

## Or manually run the code

Activate the environment:
```
conda activate mdd_env
```

### Preprocess the speechocean762 data

Get the speechocean762 data
```
git clone https://github.com/jimbozhang/speechocean762.git
mkdir data data/real_data
cp speechocean762/resource/scores.json data/real_data
```

Run the scripts to preprocess the data
```
python data_preprocessing/real_data/preprocess_speechocean762.py
python data_preprocessing/real_data/split_speechocean762.py
```

speechocean762 data:

Zhang, J., Zhang, Z., Wang, Y., Yan, Z., Song, Q., Huang, Y., Li, K., Povey, D., Wang, Y. (2021) speechocean762: An Open-Source Non-Native English Speech Corpus for Pronunciation Assessment. Proc. Interspeech 2021, 3710-3714, doi: 10.21437/Interspeech.2021-1259

https://github.com/jimbozhang/speechocean762

### Make the synthetic data

Run the scripts to get and preprocess the data
```
python data_preprocessing/synthetic_data/get_sentences.py
python data_preprocessing/synthetic_data/filter_sentences.py
python data_preprocessing/synthetic_data/make_ipa_transcriptions.py
python data_preprocessing/synthetic_data/make_pronunciation_errors.py
```

visual_genome-simple-en data:

This is the dataset of simple English sentences.

https://huggingface.co/datasets/Jotschi/visual_genome-simple-en

### Baselines

Run the majority class baseline

```
python baselines/scripts/majority_class.py
```

Run the random bias baseline

```
python baselines/scripts/random_bias.py
```

### Training

Get the vocabulary
```
python training/get_vocab.py
```

Train the model (this example uses the train split from speechocean762)
```
python training/train.py train --train data/real_data/train.json --output experiments/experiment
```

Create the loss and scores graphs

```
python training/loss_graph.py experiments/experiment/training_log.txt --output_dir experiments/experiment
```

### Evaluation

Run the evaluation pipeline on the predictions file
```
python evaluation/evaluation.py single --pred experiments/experiment/predictions.json
```
