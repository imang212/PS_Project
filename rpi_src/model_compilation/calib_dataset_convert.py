import cv2
import numpy as np
import os
from pathlib import Path

input_dir = "calib_dataset"
output_dir = "calib_dataset_npy"
img_size = 640
Path(output_dir).mkdir(parents=True, exist_ok=True)
i = 0
for file in os.listdir(input_dir):
    path = os.path.join(input_dir, file)
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cv2.imread(path)
    if img is None:
        continue
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize
    img = cv2.resize(img, (img_size, img_size)) # batch dimension (640, 640, 3)
    # float32 + normalise
    img = img.astype(np.uint8) # int type for model normalisation process (640, 640, 3)
    # add batch dimension
    #img = np.expand_dims(img, axis=0) # for batch dimension model shape - (1, 640, 640, 3)
    np.save(os.path.join(output_dir, f"{i}.npy"), img)
    i += 1
print(f"Converted {i} images to calibration dataset")