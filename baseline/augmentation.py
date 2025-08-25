"""augmentation.py

This script collects images from a specified root directory, splits them into training and test sets,
applies various augmentations to the training set, and saves the augmented images to an output directory.

The augmentations include :
    - horizontal/vertical flips,
    - rotations,
    - brightness/contrast/sharpness adjustments,
    - blurring, zooming in/out,
    - translations,
    - perspective transformations,
    - hue/saturation shifts,
    - noise addition.

The script also allows for copying original images to the output directory without augmentation.

Usage:
    Run the script to perform the augmentation process. The augmented images will be saved in the specified
    output directory.

Requirements:
    - Python 3.x
    - Pillow library for image processing
    - NumPy for numerical operations
    - tqdm for progress bars

Note:
    The script assumes the input images are in JPEG or PNG format and are organized in subfolders by class.
    The output images will be saved in JPEG format.

Args:
    root (str): The root directory containing class subfolders with images.
    output_dir (str): The directory where augmented images will be saved.

Returns:
    None: The function does not return any value. It saves the augmented images to the specified output directory.

"""

import os
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from PIL import ImageDraw
import numpy as np
from tqdm import tqdm

# ------------------
# Config
# ------------------
RAW_DATA_DIR = "raw_data"  # Root folder containing all class subfolders
OUTPUT_DIR = "augmented_data_no_pad"  # Augmented dataset output folder

TEST_ORIGINALS = {
    "non_defect": 10,
    "hole": 1,
    "lycra_cut": 1,
    "needln": 2,
    "twoply": 1,
}

PER_ORIGINAL_COUNTS = {
    "non_defect": 200,
    "hole": 200,
    "lycra_cut": 200,
    "needln": 200,
    "twoply": 200,
}

PROBS = {
    "hflip": 0.5,
    "vflip": 0.5,
    "rotate180": 0.55,
    "rotate5": 0.55,
    "brightness": 0.6,
    "contrast": 0.7,
    "sharpness": 0.6,
    "blur": 0.4,
    "zoom_out": 0.4,
    "zoom_in": 0.4,
    "translate": 0.65,
    "perspective": 0.45,
    "hue_saturation": 0.2,
    "noise": 0.2,
}


# ------------------
# Helper functions
# ------------------
def collect_images(root=RAW_DATA_DIR):
    classes = {}
    for root_dir, dirs, files in os.walk(root):
        if root_dir == root:
            continue
        cls_name = os.path.basename(root_dir)
        image_files = [
            os.path.join(root_dir, f)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if image_files:
            classes[cls_name] = image_files
    return classes


def split_originals(classes):
    """Split images into training and test sets based on TEST_ORIGINALS."""
    train_split = {}
    test_split = {}
    for cls, images in classes.items():
        random.shuffle(images)
        val_count = TEST_ORIGINALS.get(cls, 0)
        test_split[cls] = images[:val_count]
        train_split[cls] = images[val_count:]
    return train_split, test_split


def zoom_out(image, scale=0.8, fill_color=(0, 0, 0)):
    """Zoom out by scaling down and padding to original size."""
    w, h = image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.LANCZOS)
    background = Image.new("RGB", (w, h), fill_color)
    offset = ((w - new_w) // 2, (h - new_h) // 2)
    background.paste(image_resized, offset)
    return background


def zoom_in(image, scale=0.9):
    """Zoom in by cropping and resizing back."""
    w, h = image.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


def translate_image(image, max_shift=0.02):
    """Translate image by percentage of its size."""
    w, h = image.size
    shift_x = int(random.uniform(-max_shift, max_shift) * w)
    shift_y = int(random.uniform(-max_shift, max_shift) * h)
    background = Image.new("RGB", (w, h), (0, 0, 0))
    background.paste(image, (shift_x, shift_y))
    return background


def perspective_transform(image, max_warp=0.015):
    """Apply a random perspective warp."""
    w, h = image.size
    shift_w = int(w * max_warp)
    shift_h = int(h * max_warp)

    # Define source and destination quadrilaterals
    src_quad = (0, 0, w, 0, w, h, 0, h)
    dst_quad = (
        random.randint(0, shift_w),
        random.randint(0, shift_h),
        w - random.randint(0, shift_w),
        random.randint(0, shift_h),
        w - random.randint(0, shift_w),
        h - random.randint(0, shift_h),
        random.randint(0, shift_w),
        h - random.randint(0, shift_h),
    )

    # Apply warp
    return image.transform(
        (w, h), Image.QUAD, dst_quad, resample=Image.BICUBIC
    )


def hue_saturation_shift(image, hue_shift=0.05, sat_shift=0.2):
    """Randomly shift hue and saturation."""
    img = np.array(image.convert("HSV"))
    img = img.astype(np.float32)
    img[..., 0] = (img[..., 0] + hue_shift * 255 * random.uniform(-1, 1)) % 255
    img[..., 1] = np.clip(
        img[..., 1] * (1 + sat_shift * random.uniform(-1, 1)), 0, 255
    )
    img = img.astype(np.uint8)
    return Image.fromarray(img, "HSV").convert("RGB")


def add_noise(image, noise_level=0.02):
    """Add random Gaussian noise."""
    img = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, img.shape)
    img = np.clip(img + noise, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def augment_image(image):
    if random.random() < PROBS["hflip"]:
        image = ImageOps.mirror(image)
    if random.random() < PROBS["vflip"]:
        image = ImageOps.flip(image)
    if random.random() < PROBS["rotate180"]:
        image = image.rotate(180, expand=True)
    if random.random() < PROBS["rotate5"]:
        image = image.rotate(
            random.uniform(-5, 5), expand=True, fillcolor=(0, 0, 0)
        )
    if random.random() < PROBS["brightness"]:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.8, 1.2)
        )
    if random.random() < PROBS["contrast"]:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < PROBS["sharpness"]:
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < PROBS["blur"]:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))
        )
    if random.random() < PROBS["zoom_out"]:
        image = zoom_out(image, scale=random.uniform(0.9, 0.99))
    if random.random() < PROBS["zoom_in"]:
        image = zoom_in(image, scale=random.uniform(0.9, 0.99))
    if random.random() < PROBS["translate"]:
        image = translate_image(image, max_shift=0.02)
    if random.random() < PROBS["perspective"]:
        image = perspective_transform(image, max_warp=0.015)
    if random.random() < PROBS["hue_saturation"]:
        image = hue_saturation_shift(image)
    if random.random() < PROBS["noise"]:
        image = add_noise(image, noise_level=0.02)
    return image


def save_augmented(images, cls, split, count_per_original):
    save_dir = os.path.join(OUTPUT_DIR, split, cls)
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    for img_path in tqdm(images, desc=f"Augmenting {split} - {cls}"):
        img = Image.open(img_path).convert("RGB")
        for _ in range(count_per_original):
            aug_img = augment_image(img)
            # aug_img = pad_to_square(aug_img)
            aug_img.save(os.path.join(save_dir, f"{cls}_{split}_{idx}.jpeg"))
            idx += 1


def copy_originals(images, cls, split):
    save_dir = os.path.join(OUTPUT_DIR, split, cls)
    os.makedirs(save_dir, exist_ok=True)
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        # img = pad_to_square(img)
        img.save(os.path.join(save_dir, os.path.basename(img_path)))


# ------------------
# Main
# ------------------
def main():
    random.seed(42)
    classes = collect_images(RAW_DATA_DIR)
    print("\nFound images per class:")
    for cls, imgs in classes.items():
        print(f"{cls}: {len(imgs)}")
    train_split, val_split = split_originals(classes)
    for cls, images in train_split.items():
        count_per_original = PER_ORIGINAL_COUNTS.get(cls, 1)
        save_augmented(images, cls, "train", count_per_original)
        copy_originals(images, cls, "train")
    for cls, images in val_split.items():
        count_per_original = PER_ORIGINAL_COUNTS.get(cls, 1)
        save_augmented(images, cls, "test", count_per_original)
        copy_originals(images, cls, "test")
    print("\n Augmentation complete. Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
