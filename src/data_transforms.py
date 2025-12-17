from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

# ── Base pipeline (resize + normalize) ───────────────────────────────────
base_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── Training pipeline (augment + base) ───────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── Validation / Test pipelines ──────────────────────────────────────────
val_transforms  = base_transforms
test_transforms = base_transforms
