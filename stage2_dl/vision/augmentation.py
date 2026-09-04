import torchvision.transforms as T

def get_train_transforms(rotation_degrees=10, translation=(0.05, 0.05), use_hflip=False):
    """
    Returns training augmentation pipeline.
    Conservative settings recommended for medical imaging.
    """
    transforms_list = [
        T.RandomRotation(degrees=rotation_degrees),
        T.RandomAffine(degrees=0, translate=translation)
    ]
    if use_hflip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        
    return T.Compose(transforms_list)

def get_val_test_transforms():
    """
    Validation and test must use deterministic preprocessing only.
    No augmentation.
    """
    return None
