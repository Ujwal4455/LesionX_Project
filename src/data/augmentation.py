import albumentations as A

def transforms(size=384,train=False):
    ops=[A.Resize(size,size)]
    if train: ops=[A.RandomResizedCrop(size,size,scale=(.85,1.0)),A.HorizontalFlip(p=.5),A.VerticalFlip(p=.2),A.Rotate(limit=20,p=.4),A.RandomBrightnessContrast(p=.3)]+ops
    return A.Compose(ops)
