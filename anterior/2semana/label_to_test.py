import os
import json
import random

labels = '/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_labels.csv'
new_labels20 = '/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_labels20.csv'
new_labels80 = '/home/jrola/PycharmProjects/pytorch_CTM/hmdb4_labels80.csv'
class_name_to_label_path = '/home/jrola/PycharmProjects/pytorch_CTM/class_name_to_label.json'

random.seed()

if os.path.isfile(labels) == False:
    print("No File")
    exit()

with open(class_name_to_label_path, 'r') as json_file:
    x = json.load(json_file)

count = 0
lineC = 0
suffixCount = 0
suffixSum = 0
count80 = 0
count20 = 0
numLabels = int(input("Number of labels: "))

for (k, v) in x.items():
    with open(labels, 'r') as fp:
        lineC = 0
        suffixCount = 0
        for line in fp:
            lineC = lineC + 1
            if line.endswith("," + str(v) + "\n"):
                suffixCount = suffixCount + 1
                if random.random() > 0.80:
                    with open(new_labels20, 'a') as fp2:
                        fp2.write(line)
                        count20 = count20 + 1
                else:
                    with open(new_labels80, 'a') as fp3:
                        fp3.write(line)
                        count80 = count80 + 1
    print(k + " " + str(suffixCount))
    count = count + 1
    suffixSum = suffixSum + suffixCount
    if v == (numLabels - 1):
        break
countSum = count80 + count20
#print(suffixCount)
print(lineC - 1)
print(suffixSum)
print(countSum)
