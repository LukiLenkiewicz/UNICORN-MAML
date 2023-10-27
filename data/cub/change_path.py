import csv
import os

cwd = os.getcwd()
SOURCE_DIR = "split-old"
TARGET_DIR = "split"
NEW_PATH = cwd + "/images"
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"
VAL_FILENAME = "val.csv"

def change_path(filename, source=SOURCE_DIR, target=TARGET_DIR):
    with open(f'{source}/{filename}', newline='') as csvfile, open(f'{target}/{filename}', mode="w", newline='') as newcsv:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        spamwriter = csv.writer(newcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(spamreader):
            if i > 0:
                new_image_path = row[0]
                new_image_path = new_image_path.split("/")
                new_image_path = NEW_PATH + "/" + "/".join(new_image_path[-2:])
                new_row = [new_image_path] + [row[1]] + row[2:]
                spamwriter.writerow(new_row)
            else:
                spamwriter.writerow(row)

change_path(TEST_FILENAME)
change_path(TRAIN_FILENAME)
change_path(VAL_FILENAME)
