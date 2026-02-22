# Synthetic Data for Mispronunciation Detection

## Setup

```
git clone https://github.com/ella84211/synthetic-mispronunciation-data.git
cd synthetic-mispronunciation-data
pip install -r requirements.txt
```

## Preprocessing speechocean762 data
Get the speechocean762 data
```
git clone https://github.com/jimbozhang/speechocean762.git
cp speechocean762/resource/scores.json real_data
```

Run the scripts to preprocess the data
```
python real_data/preprocess_speechocean762.py
python real_data/split_speechocean762.py
```

### speechocean762 data:

Zhang, J., Zhang, Z., Wang, Y., Yan, Z., Song, Q., Huang, Y., Li, K., Povey, D., Wang, Y. (2021) speechocean762: An Open-Source Non-Native English Speech Corpus for Pronunciation Assessment. Proc. Interspeech 2021, 3710-3714, doi: 10.21437/Interspeech.2021-1259

https://github.com/jimbozhang/speechocean762

## Making the synthetic data
Run the scripts to get and preprocess the data
```
python synthetic_data/get_sentences.py
python synthetic_data/make_ipa_transcriptions.py
python synthetic_data/make_pronunciation_errors.py
python synthetic_data/split_synthetic_data.py
```