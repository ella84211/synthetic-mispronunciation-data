import copy


def format_dataset(dataset):
    formatted_dataset = []
    for sample in dataset:
        formatted_sample = format_sample(sample)
        if formatted_sample:
            formatted_dataset.append(formatted_sample)
    return formatted_dataset


def format_sample(data_sample):
    sample = copy.deepcopy(data_sample)
    required_keys = ["phones", "mispronunciations"]
    if "words" not in sample: return None
    word_errors = [0 for _ in range(len(sample["words"]))]
    phone_errors_by_word = [[0 for _ in range(len(word["phones"]))] for word in sample["words"]]
    for word in sample["words"]:
        if any(key not in word for key in required_keys): return None
    for i, word in enumerate(sample["words"]):
        reference = word["phones"]
        if word["mispronunciations"]:
            word_errors[i] = len(word["mispronunciations"])
            reference = word["phones"][:]
            pronounced = reference[:]
            deletions = []
            for error in sorted(word["mispronunciations"], key=lambda e: e["index"]):
                if error["pronounced"] == "<DEL>":
                    pronounced.pop(error["index"]-len(deletions))
                    deletions.append(error["index"])
                else:
                    phone_errors_by_word[i][error["index"]] = 1
                    pronounced[error["index"]-len(deletions)] = error["pronounced"]
            phone_errors_by_word[i] = [error for j, error in enumerate(phone_errors_by_word[i]) if j not in deletions]
            word["pronounced"] = pronounced
        else:
            word["pronounced"] = reference
        word["reference"] = reference
    sample["pronounced"] = sample["words"][0]["pronounced"][:]
    sample["reference"] = sample["words"][0]["reference"][:]
    for word in sample["words"][1:]:
        sample["pronounced"].append("<|>")
        sample["reference"].append("<|>")
        sample["pronounced"] += word["pronounced"]
        sample["reference"] += word["reference"]
    phone_errors = phone_errors_by_word[0]
    for word in phone_errors_by_word[1:]:
        phone_errors.append("<|>")
        phone_errors += word
    sample["sentence errors"] = sum(word_errors)
    sample["word errors"] = word_errors
    sample["phone errors"] = phone_errors
    return sample