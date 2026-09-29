import albumentations as A


def build_train_augmentations(image_size=384):
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.HueSaturationValue(p=0.2),
        A.Resize(image_size, image_size),
    ])


def build_val_augmentations(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
