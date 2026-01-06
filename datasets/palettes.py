import numpy as np

# WHDLD Dataset Palette(6 classes)
WHDLD_PALETTE = np.array([
    [255, 0, 0],    # building
    [255, 255, 0],  # road
    [192, 192, 0],  # pavement
    [0, 255, 0],    # vegetation
    [128, 128, 128],# bare soil
    [0, 0, 255],    # water
], dtype=np.uint8)

# LoveDA Dataset Palette (7 classes)
LOVEDA_PALETTE = np.array([
    [255, 255, 255],  # Background
    [255, 0, 0],      # Building
    [255, 255, 0],    # Road
    [0, 0, 255],      # Water
    [159, 129, 183],  # Barren
    [0, 255, 0],      # Forest
    [255, 195, 128],  # AgriculturaSl
], dtype=np.uint8)

# Potsdam Dataset Palette (6 classes)
POTSDAM_PALETTE = np.array([
    [255, 255, 255],  # Impervious_surface
    [0, 0, 255],      # Building 
    [0, 255, 255],    # Low_vegetation
    [0, 255, 0],      # Tree
    [255, 255, 0],    # Car
    [255, 0, 0],      # Clutter
], dtype=np.uint8)

# GID-15 Dataset Palette (15 classes)
GID_PALETTE = np.array([
    [200, 0, 0], [250, 0, 150], [200, 150, 150], [250, 150, 150], [0, 200, 0],
    [150, 250, 0], [150, 200, 150], [200, 0, 200], [150, 0, 250], [150, 150, 250],
    [250, 200, 0], [200, 200, 0], [0, 0, 200], [0, 150, 200], [0, 200, 250]   
], dtype=np.uint8)

# MER/MSL Dataset Palette (9 classes)
MER_PALETTE = np.array([ 
    [128, 0, 0], [0, 128, 0], [128, 128, 0], 
    [0, 0, 128], [128, 0, 128],[0, 128, 128], 
    [128, 128, 128], [64, 0, 0], [192, 0, 0],
], dtype=np.uint8)


def get_palette(dataset):
    if dataset == 'whdld':
        return WHDLD_PALETTE
    elif dataset == 'loveda':
        return LOVEDA_PALETTE
    elif dataset == 'potsdam':
        return POTSDAM_PALETTE
    elif dataset == 'gid':
        return GID_PALETTE
    elif dataset == 'mer' or 'msl':
        return MER_PALETTE
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    