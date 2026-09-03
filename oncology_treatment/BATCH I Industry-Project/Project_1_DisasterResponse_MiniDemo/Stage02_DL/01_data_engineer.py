"""
ROLE: DATA ENGINEER (Deep Learning stage)
JOB: "Label street footage; perform data augmentation for night and heavy
rain conditions."

WHY THIS STEP EXISTS:
Real photos aren't available for this classroom demo, so we GENERATE tiny
8x8-pixel synthetic "images" that behave the same way real ones would:
flooded street patches are brighter (water reflects light/camera flash) in
the lower half, clear street patches are darker/uniform. This keeps the
workflow identical to the real project, just at toy scale.
"""

import numpy as np

np.random.seed(42)


def make_image(flooded: bool):
    """Creates one tiny 8x8 grayscale 'photo'. flooded=True -> brighter lower half."""
    img = np.random.uniform(0.1, 0.3, size=(8, 8))  # dark road background
    if flooded:
        img[4:, :] += np.random.uniform(0.4, 0.6, size=(4, 8))  # bright reflective water
    return np.clip(img, 0, 1)


# STEP 1: Generate a small labeled set of images (8 flooded, 8 clear)
images, labels = [], []
for _ in range(8):
    images.append(make_image(flooded=True))
    labels.append("Flooded")
for _ in range(8):
    images.append(make_image(flooded=False))
    labels.append("Clear")

images = np.array(images)
labels = np.array(labels)

# STEP 2: Simple data AUGMENTATION - simulate "night" conditions by darkening
# a copy of every image (this is what the problem statement asks for:
# "augmentation for night and heavy rain conditions")
night_images = images * 0.5  # darker versions
images = np.concatenate([images, night_images])
labels = np.concatenate([labels, labels])

# STEP 3: Save as flattened rows (64 pixel values) + label, ready for training
flat = images.reshape(len(images), -1)  # 8x8 -> 64 values per image
np.save("data/images.npy", flat)
np.save("data/labels.npy", labels)

print(f"Generated and labeled {len(images)} mini images (with night augmentation).")
print(f"Class counts: Flooded={sum(labels=='Flooded')}, Clear={sum(labels=='Clear')}")
