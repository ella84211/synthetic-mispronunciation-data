import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns
import numpy as np
import math

INPUT_FILE = "real_data/real_transcriptions.json"
# get this for train, val, and test
INPUT_FILE = "real_data/validate.json"
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

samples_with_errors = 0
words_with_errors = 0
errors = 0
list_of_words_with_errors = []

words = 0

for sample in data:
    # Counting samples and words with errors
    # Also errors in words that have errors
    error = False
    for word in sample["words"]:
        if word["mispronunciations"]:
            error = True
            words_with_errors += 1
            errors += len(word["mispronunciations"])
            list_of_words_with_errors.append(word)
    if error: samples_with_errors += 1

    # Counting total words
    words += len(sample["words"])

    

print("Samples with errors:", samples_with_errors)
print("Words with errors:", words_with_errors)
print("Average errors per word with 1+ errors:", errors/words_with_errors)
print("Average words per sentence:", words/len(data))
print("Average erroneous words in a sentence with errors:", words_with_errors/samples_with_errors)

with open("output.json", "w", encoding="utf-8") as file:
    json.dump(list_of_words_with_errors, file, indent=2, ensure_ascii=False)

unk_count = 0
del_count = 0
errors = 0

# Getting error patterns (which letters are substituted?) into a count dictionary
substitutions = {}
for word in list_of_words_with_errors:
    for error in word["mispronunciations"]:
        errors += 1
        if error["pronounced"] == "<UNK>":
            unk_count += 1
        if error["pronounced"] == "<DEL>":
            del_count += 1
        substitution = (error["canonical"], error["pronounced"])
        if substitution in substitutions: substitutions[substitution] += 1
        else: substitutions[substitution] = 1
print("Unknowns:", unk_count, errors, unk_count/errors)
print("Deletions:", del_count, errors, del_count/errors)
reversed_sub_counts = {}
for entry in substitutions:
    if substitutions[entry] in reversed_sub_counts: reversed_sub_counts[substitutions[entry]].append(entry)
    else: reversed_sub_counts[substitutions[entry]] = [entry]
with open("output2.tsv", "w", encoding="utf-8") as file:
    file.write("len(list)\t#instances\t[(expected, pronounced)]\n")
    for entry in sorted(reversed_sub_counts.keys()):
        file.write(str(len(reversed_sub_counts[entry])) + "\t" + str(entry) + "\t" + str(reversed_sub_counts[entry]) + "\n")

words = [word["text"] for word in list_of_words_with_errors]
word_counts = {}
for word in words:
    if word in word_counts: word_counts[word] += 1
    else: word_counts[word] = 1
reversed_word_counts = {}
for word in word_counts:
    if word_counts[word] in reversed_word_counts: reversed_word_counts[word_counts[word]].append(word)
    else: reversed_word_counts[word_counts[word]] = [word]
with open("output1.tsv", "w", encoding="utf-8") as file:
    file.write("len(list)\t#instances\t['word']\n")
    for word in sorted(reversed_word_counts.keys()):
        file.write(str(len(reversed_word_counts[word])) + "\t" + str(word) + "\t" + str(reversed_word_counts[word]) + "\n")














# figures

phones = {}

for sample in data:
    for word in sample["words"]:
        for phone in word["phones"]:
            if phone not in phones: phones[phone] = {"occurences": 0, "mispronounced": 0, "replaced by": {}}
            phones[phone]["occurences"] += 1
        if word["mispronunciations"]:
            for error in word["mispronunciations"]:
                phones[error["canonical"]]["mispronounced"] += 1
                if error["pronounced"] not in phones[error["canonical"]]["replaced by"]:
                    phones[error["canonical"]]["replaced by"][error["pronounced"]] = 0
                phones[error["canonical"]]["replaced by"][error["pronounced"]] += 1
"""for phone in phones:
    print(f"{phone}\t{phones[phone]}")"""

# Collect all substitution pairs across phonemes
all_subs = Counter()
for phoneme, info in phones.items():
    for sub, count in info["replaced by"].items():
        pair = f"{phoneme} → {sub}"  # label each substitution
        all_subs[pair] += count

# Take top N substitutions
top_n = 30
top_subs = all_subs.most_common(top_n)
labels, counts = zip(*top_subs)

plt.rcParams["font.family"] = "Times New Roman"

# Plot
plt.figure(figsize=(12,6))
plt.bar(labels, counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Occurrences")
#plt.title(f"Top {top_n} Mispronounced Substitutions Across All Phonemes")
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
#plt.savefig("figures/barchart.pdf", format="pdf")











df = pd.read_csv("figures/phoneme_annotations_all.csv")
#df = pd.read_csv("figures/phoneme_annotations.csv")

df = df[df['canonical_phoneme'] != df['produced_phoneme']]

sub_counts = df.groupby(['canonical_phoneme', 'produced_phoneme']).size().reset_index(name='count')

total_counts = df.groupby('canonical_phoneme').size().reset_index(name='total')
sub_counts = sub_counts.merge(total_counts, on='canonical_phoneme')
sub_counts['proportion'] = sub_counts['count'] / sub_counts['total']

heatmap_data = sub_counts.pivot(index='produced_phoneme', columns='canonical_phoneme', values='proportion').fillna(0)

plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(10, 12))
sns.heatmap(heatmap_data, annot=False, cmap="Blues", yticklabels=True, xticklabels=True, linewidths=0.5, fmt=".2f")
plt.xlabel("Intended Phoneme")
plt.ylabel("Pronounced Phoneme")
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)










plt.figure()
word_scores = []
sentence_scores = []

for sample in data:
    sentence_scores.append(sample["accuracy"])
    for word in sample["words"]:
        word_scores.append(word["accuracy"])

plt.rcParams["font.family"] = "Times New Roman"

fig, axes = plt.subplots(2, 1, sharex=True)

# Word-level histogram
axes[0].hist(word_scores, bins=range(0, 12), align='left', color='mediumturquoise', edgecolor='black')
axes[0].set_ylabel("Occurences")
axes[0].text(0.02, 0.95, "(a)", transform=axes[0].transAxes,
             fontsize=12, fontweight="bold", va="top")

#axes[0].grid(axis='y', alpha=0.7)

# Sentence-level histogram
axes[1].hist(sentence_scores, bins=range(0, 12), align='left', color='skyblue', edgecolor='black')
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Occurences")
axes[1].text(0.02, 0.95, "(b)", transform=axes[1].transAxes,
             fontsize=12, fontweight="bold", va="top")

#axes[1].grid(axis='y', alpha=0.7)

# Shared x-axis
axes[1].set_xticks(range(0, 11))

plt.tight_layout()











plt.figure()
num_bins = 10

bins = [0 for _ in range(num_bins)]

for sample in data:
    for word in sample["words"]:
        if word["mispronunciations"]:
            for error in word["mispronunciations"]:
                position = error["index"] / (len(word["phones"])-1) if len(word["phones"]) > 1 else 0.5
                bin_index = min(int(position * num_bins), num_bins-1)
                bins[bin_index] += 1 # this should be inside the loop

error_rate_per_bin = [bins[i]/sum(bins) if bins[i] > 0 else 0 for i in range(num_bins)]
bin_centers = [(i/num_bins + (i+1)/num_bins)/2 for i in range(num_bins)]

plt.rcParams["font.family"] = "Times New Roman"

plt.plot(bin_centers, error_rate_per_bin, marker='o', linestyle='-', color="mediumturquoise")#, markerfacecolor='blue', markeredgecolor='black')
plt.xlabel("Relative Phoneme Position in Word")
plt.ylabel("Mispronunciation Rate")
#plt.title("Positional Phoneme Error Rate (PPER) - Aggregate Across All Phonemes")
plt.grid(True)
plt.ylim(0, 1)











plt.figure()
num_bins = 10
errors = 0

bins = [0 for _ in range(num_bins)]

for sample in data:
    for word in sample["words"]:
        if word["mispronunciations"]:
            for error in word["mispronunciations"]:
                #if error["pronounced"] != "<DEL>": continue
                errors += 1
                position = (error["index"]+1) / (len(word["phones"])) if len(word["phones"]) > 1 else 0.5
                bin_index = min(int(position * num_bins), num_bins-1)
                bins[bin_index] += 1

print("Errors:", errors)
error_rate_per_bin = [bins[i]/sum(bins) if bins[i] > 0 else 0 for i in range(num_bins)]
bin_centers = [(i/num_bins + (i+1)/num_bins)/2 for i in range(num_bins)]

plt.rcParams["font.family"] = "Times New Roman"

plt.plot(bin_centers, error_rate_per_bin, marker='o', linestyle='-', color="mediumturquoise")#, markerfacecolor='blue', markeredgecolor='black')
plt.xlabel("Relative Phoneme Position in Word")
plt.ylabel("Mispronunciation Rate")
plt.xticks(np.linspace(0, 1, 11))
#plt.title("Positional Phoneme Error Rate (PPER) - Aggregate Across All Phonemes")
plt.grid(True)
plt.ylim(0, 1)













plt.figure()
phones = {}

for sample in data:
    for word in sample["words"]:
        for phone in word["phones"]:
            if phone not in phones: phones[phone] = {"occurences": 0, "mispronounced": 0, "replaced by": {}}
            phones[phone]["occurences"] += 1
        if word["mispronunciations"]:
            for error in word["mispronunciations"]:
                phones[error["canonical"]]["mispronounced"] += 1
                if error["pronounced"] not in phones[error["canonical"]]["replaced by"]:
                    phones[error["canonical"]]["replaced by"][error["pronounced"]] = 0
                phones[error["canonical"]]["replaced by"][error["pronounced"]] += 1

list_phones = []
entropies = []
error_rates = []
for phone in phones:
    list_phones.append(phone)
    error_rates.append(phones[phone]["mispronounced"]/phones[phone]["occurences"])
    total_errors = phones[phone]["mispronounced"]
    if total_errors == 0:
        entropies.append(0)
        continue
    entropy = -sum((count/total_errors) * math.log2(count/total_errors) for count in phones[phone]["replaced by"].values())
    entropies.append(entropy)

plt.rcParams["font.family"] = "Times New Roman"

plt.scatter(error_rates, entropies, c="deepskyblue")
plt.xlabel("Error rates")
plt.ylabel("Entropy")
#plt.title("Error Rates vs. Entropies")
plt.grid(True)
#plt.savefig("figures/scatter.pdf", format="pdf")










plt.figure()
counts = {}

for sample in data:
    words = len(sample["words"])
    if words not in counts:
        counts[words] = 0
    counts[words] += 1
    """if words == 2:
        print(sample["text"])"""

plt.rcParams["font.family"] = "Times New Roman"

print(counts)

plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.ylabel("Occurences")
plt.xlabel("Words in sentence")
plt.xticks(np.linspace(0, 10, 11))












phones = {}

for sample in data:
    for word in sample["words"]:
        for phone in word["phones"]:
            if phone not in phones: phones[phone] = 0
            phones[phone] += 1

counts = {}
for phone in phones:
    if phones[phone] not in counts: counts[phones[phone]] = []
    counts[phones[phone]].append(phone)

phones = {}
for count in sorted(counts.keys(), reverse=True):
    for phone in counts[count]:
        phones[phone] = count

plt.rcParams["font.family"] = "Times New Roman"

# Plot
plt.figure(figsize=(12,6))
plt.bar(phones.keys(), phones.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Occurrences")
#plt.title(f"Top {top_n} Mispronounced Substitutions Across All Phonemes")
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()











phones = {}
errors = {}

for sample in data:
    for word in sample["words"]:
        if word["mispronunciations"]:
            for phone in word["mispronunciations"]:
                if phone["canonical"] not in phones: phones[phone["canonical"]] = 0
                phones[phone["canonical"]] += 1
                if phone["canonical"] == "ʌ":
                    if phone["pronounced"] not in errors:
                        errors[phone["pronounced"]] = 0
                    errors[phone["pronounced"]] += 1

print(errors)
counts = {}
for phone in phones:
    if phones[phone] not in counts: counts[phones[phone]] = []
    counts[phones[phone]].append(phone)

phones = {}
for count in sorted(counts.keys(), reverse=True):
    for phone in counts[count]:
        phones[phone] = count

plt.rcParams["font.family"] = "Times New Roman"

# Plot
plt.figure(figsize=(12,6))
plt.bar(phones.keys(), phones.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Occurrences")
#plt.title(f"Top {top_n} Mispronounced Substitutions Across All Phonemes")
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()










plt.show()