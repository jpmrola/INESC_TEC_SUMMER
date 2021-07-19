import cv2
import os
import json

def extract_frames(file_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(target_dir, f'{count}.jpg')
        cv2.imwrite(frame_path, image)     # save frame as JPEG file
        success, image = vidcap.read()
        success, image = vidcap.read()
        count += 1


root = '/home/jrola/PycharmProjects/pytorch_CTM/data/hmdb4_org'
frame_root = '/home/jrola/PycharmProjects/pytorch_CTM/data/hmdb4_frames'
labels_path = '/home/jrola/PycharmProjects/pytorch_CTM/data/hmdb4_labels.csv'
class_name_to_label_path = '/home/jrola/PycharmProjects/pytorch_CTM/data/class_name_to_label_4.json'

# read files
files = []

for class_name in os.listdir(root):
    for video_name in os.listdir(os.path.join(root, class_name)):
        files.append([os.path.join(class_name, video_name), class_name])

# normalize labels
class_name_to_label = {}
current_label = -1

for vid in files:
    label = class_name_to_label.get(vid[1], -1)

    if label == -1:
        current_label += 1
        class_name_to_label[vid[1]] = current_label
        label = current_label

    vid[1] = label


# save file paths
if not os.path.exists(os.path.split(labels_path)[0]):
    os.makedirs(os.path.split(labels_path)[0])

f = open(labels_path, 'w')

f.write('path,label\n')

for vid in files:
    f.write(f'{vid[0]},{vid[1]}\n')

f.close()

# save label normalization
if not os.path.exists(os.path.split(class_name_to_label_path)[0]):
    os.makedirs(os.path.split(class_name_to_label_path)[0])

with open(class_name_to_label_path, 'w') as json_file:
    json.dump(class_name_to_label, json_file, indent=4)

# extract frames
for i, vid in enumerate(files):
    file_path = os.path.join(root, vid[0])
    target_dir = os.path.join(frame_root, vid[0])

    extract_frames(file_path, target_dir)

    print(f'{i+1}/{len(files)}')
