from collections import Counter
import json


def load_file(filepath="/home/lukasz/UNICORN-MAML/data/emnist/novel.json"):
    with open(filepath, "r") as f:
        content = json.load(f)
    return content


def change_labels(labels):
    new_labels = []
    current_label = labels[0]
    current_new_label = 0
    for label in labels:
        if label != current_label:
            current_new_label += 1
            current_label = label

        new_labels.append(current_new_label)

    return new_labels


def save_file(content, filepath="/home/lukasz/UNICORN-MAML/data/emnist/novel.json"):
    with open(filepath, "w") as f:
        json.dump(content, f)


file = load_file()
new_labels = change_labels(file["image_labels"])
file["image_labels"] = new_labels

print(json.dumps(file))

save_file(file)
