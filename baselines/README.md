# Baselines

There are two baselines: majority class and random bias

## Majority class

Because this is primarily a classification task (to classify whether a phone, word, or sentence has an error), the first baseline is majority class. The majority of phones, words, and sentences are pronounced correctly, so 0 (no error) is the majority class. Therefore, this baseline predicts every phone as correct. It has phone and word F1 scores of 0, because it predicts no errors.

## Random bias

The random bias baseline randomly predicts errors based on frequencies observed in the speechocean762 data. The default probabilities for whether a sentence has an error are: [0.8, 0.1, 0.05, 0.025, 0.025]. This means that a sentence has 0.8 chance of having no errors, 0.1 chance of having one error, 0.05 chance of having two errors, etc. The positions of the errors in the sentences is random, with no weights. The default probability that an error is a deletion is 0.25. This means that once the position of an error is randomly selected, there is 0.25 chance that it is a deletion instead of a substitution.