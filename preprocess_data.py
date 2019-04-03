# File: preprocess_data.py
# Author: Ronil Pancholia
# Date: 4/6/19
# Time: 5:26 AM

import os
import time

from PIL import Image

from face_annotations import Face_Annotations

data_dir = "data/originalPics"
annotations_dir = "data/FDDB-folds"
face_out_dir = "data/processed/face"
no_face_out_dir = "data/processed/no_face"

if not os.path.exists(no_face_out_dir):
    os.makedirs(no_face_out_dir)

if not os.path.exists(face_out_dir):
    os.makedirs(face_out_dir)

file_paths = []
num_faces = []
faces_annotations = []

for file_name in os.listdir(annotations_dir):
    if file_name.endswith("ellipseList.txt"):
        with open(os.path.join(annotations_dir, file_name)) as f:
            while True:
                fname = f.readline()
                if not fname:
                    break
                file_paths.append(os.path.join(data_dir, fname.strip() + ".jpg"))
                img_faces = int(f.readline().strip())
                num_faces.append(img_faces)
                for i in range(img_faces):
                    l = f.readline().strip().split()
                    faces_annotations.append(Face_Annotations(int(float(l[3])), int(float(l[4])), int(float(l[0])), int(float(l[1]))))


j=0
for i, file_path in enumerate(file_paths):
    no_face_saved = False
    for f in range(num_faces[i]):
        face_annotation = faces_annotations[j]
        left, upper, right, bottom = face_annotation.process_coordinates()
        img = Image.open(file_path)
        face_part = img.crop((left, upper, right, bottom))
        j+=1
        face_part_new = face_part.resize((60, 60))
        file_name = str(time.time()) + "_" + str(file_path.split("/")[-1])
        face_part_new.save(os.path.join(face_out_dir, file_name), "JPEG", optimize=True)
        if not no_face_saved:
            no_face_part = img.crop((0,0,60,60))
            no_face_part.save(os.path.join(no_face_out_dir, file_name), "JPEG", optimize=True)
            no_face_saved = True