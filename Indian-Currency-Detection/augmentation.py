import albumentations as A
import os
import os
import tensorflow as tf

input_folder = r"F:\currency inda2\Train"
output_folder = r"F:\currency inda2\Train\albumentation_image"
os.makedirs(output_folder, exist_ok=True)


transform = A.Compose([

    # Flip
    # # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.3),

    # # Rotation
    # A.Rotate(limit=15, p=0.4),

    # # Random Crop and Resize
    # A.RandomResizedCrop(
    #     size=(128, 128),
    #     scale=(0.8, 1.0),
    #     ratio=(0.75, 1.33),
    #     p=0.5
    # ),

    # Brightness and Contrast
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5
    ),

    # Color
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.3
    ),

    # Blur
    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.25
    ),

    # Noise
    A.GaussNoise(
        std_range=(0.02, 0.06),
        p=0.3
    ),

    # Sharpen
    # A.Sharpen(
    #     alpha=(0.2, 0.5),
    #     lightness=(0.5, 1.0),
    #     p=0.3
    # ),

    # CLAHE (improves local contrast)
    A.CLAHE(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        p=0.2
    ),

    # Affine Transform
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.1, 0.1),
        rotate=(-20, 20),
        shear=(-8, 8),
        p=0.5
    ),

    # Elastic Distortion
    # A.ElasticTransform(
    #     p=0.2
    # ),

    # Normalize image size
    A.Resize(128, 128)

])

# ==========================
# Loop through class folders
# ==========================
import cv2
classes = ["10", "20","50","100","200","500","2000","Invalid"]

for cls in classes:

    input_class = os.path.join(input_folder, cls)
    output_class = os.path.join(output_folder, cls)

    os.makedirs(output_class, exist_ok=True)

    for filename in os.listdir(input_class):

        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".avif")):

            image_path = os.path.join(input_class, filename)

            image = cv2.imread(image_path)

            if image is None:
                print("Could not read:", image_path)
                continue

            # Resize image
            image = cv2.resize(image, (224,224))

            # Save resized original
            cv2.imwrite(
                os.path.join(output_class, filename),
                image
            )

            name, ext = os.path.splitext(filename)

            # Generate augmented images
            for i in range(5):

                augmented = transform(image=image)

                aug_image = augmented["image"]

                save_path = os.path.join(
                    output_class,
                    f"{name}_aug_{i+1}{ext}"
                )

                cv2.imwrite(save_path, aug_image)

print("Augmentation Completed Successfully!")

import os
import random
import shutil

# ==========================
# Configuration
# ==========================

SOURCE_DIR = os.path.join(output_folder)
DEST_DIR = r"F:\currency inda2\Train\Train_Balanced"

# Number of images per class
TARGET_IMAGES = 12000

random.seed(42)

# Image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ==========================
# Create destination folder
# ==========================

os.makedirs(DEST_DIR, exist_ok=True)

# ==========================
# Balance dataset
# ==========================

for class_name in sorted(os.listdir(SOURCE_DIR)):

    source_class = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(source_class):
        continue

    destination_class = os.path.join(DEST_DIR, class_name)
    os.makedirs(destination_class, exist_ok=True)

    images = [
        img for img in os.listdir(source_class)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    total = len(images)

    print(f"\nClass : {class_name}")
    print(f"Original Images : {total}")

    # --------------------------
    # Case 1 : Reduce images
    # --------------------------
    if total > TARGET_IMAGES:

        selected = random.sample(images, TARGET_IMAGES)

    # --------------------------
    # Case 2 : Increase images
    # --------------------------
    elif total < TARGET_IMAGES:

        selected = images.copy()

        additional = random.choices(
            images,
            k=TARGET_IMAGES-total
        )

        selected.extend(additional)

    else:
        selected = images

    # --------------------------
    # Copy images
    # --------------------------

    filename_count = {}

    for img in selected:

        src = os.path.join(source_class, img)

        # Handle duplicated filenames
        if img in filename_count:

            filename_count[img] += 1

            name, ext = os.path.splitext(img)

            new_name = f"{name}_copy{filename_count[img]}{ext}"

        else:

            filename_count[img] = 0
            new_name = img

        dst = os.path.join(destination_class, new_name)

        shutil.copy2(src, dst)

    print(f"Balanced Images : {len(selected)}")

print("\nDataset balancing completed successfully.")




train_dataset = tf.keras.utils.image_dataset_from_directory(
    output_folder,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    shuffle = True,
    
)

val_test_dataset = tf.keras.utils.image_dataset_from_directory(
    output_folder,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

print("Training Images:", len(train_dataset.file_paths))
print("Validation Images:", len(val_test_dataset.file_paths))
class_names = train_dataset.class_names
print(class_names)  # verify actual order before trusting anything


val_batches = tf.data.experimental.cardinality(val_test_dataset)

validation_dataset = val_test_dataset.take(val_batches // 2)
test_dataset = val_test_dataset.skip(val_batches // 2)




train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x,y: (normalization_layer(x),y))
validation_dataset = validation_dataset.map(lambda x,y: (normalization_layer(x),y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))


