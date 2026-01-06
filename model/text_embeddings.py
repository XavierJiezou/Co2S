import torch
import os
import numpy as np
import clip
import argparse

# =========================
# 1. WHDLD
# =========================
WHDLD_classes = ['building', 'road', 'pavement', 'vegetation', 'bare soil', 'water']
WHDLD_classes_w_concepts = [
    ['building', 'residential building', 'commercial building', 'industrial building', 'high-rise', 'low-rise'],
    ['road', 'highway', 'arterial road', 'residential road', 'asphalt road', 'concrete road'],
    ['pavement', 'sidewalk', 'pedestrian path', 'concrete pavement', 'brick pavement', 'stone pavement'],
    ['vegetation', 'tree', 'shrub', 'grassland', 'farmland', 'vegetation patch'],
    ['bare soil', 'unvegetated land', 'soil exposure', 'construction site', 'quarry', 'desert area'],
    ['water', 'lake', 'river', 'pond', 'water body', 'water surface', 'reservoir']
]

# =========================
# 2. LOVEDA
# =========================
LOVEDA_classes = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']
LOVEDA_classes_w_concepts = [
    ['background', 'undeveloped land', 'open space', 'background area', 'unclassified region'],
    ['building', 'residential building', 'commercial building', 'industrial building', 'public building', 'high-rise'],
    ['road', 'highway', 'urban road', 'rural road', 'asphalt road', 'concrete road'],
    ['water', 'lake', 'river', 'pond', 'reservoir', 'water body'],
    ['barren', 'bare soil', 'desert', 'construction site', 'mining area', 'exposed rock'],
    ['forest', 'deciduous forest', 'evergreen forest', 'mixed forest', 'woodland', 'tree canopy'],
    ['agriculture', 'farmland', 'cropland', 'orchard', 'vineyard', 'greenhouse']
]

# =========================
# 3. Potsdam
# =========================
Potsdam_classes = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']
Potsdam_classes_w_concepts = [
    ['impervious_surface', 'asphalt', 'concrete', 'pavement', 'roof', 'sidewalk'],
    ['building', 'residential building', 'commercial building', 'church', 'factory', 'historical building'],
    ['low_vegetation', 'grass', 'lawn', 'meadow', 'pasture', 'flower bed'],
    ['tree', 'deciduous tree', 'evergreen tree', 'tree row', 'park tree', 'forest edge'],
    ['car', 'passenger car', 'truck', 'van', 'bus', 'parked vehicle'],
    ['clutter', 'fence', 'wall', 'railway', 'construction site', 'unclassified object']
]

# =========================
# 4. GID-15
# =========================
GID_classes = [
    'industrial_land', 'urban_residential', 'rural_residential', 'traffic_land',
    'paddy_field', 'irrigated_land', 'dry_cropland', 'garden_plot',
    'arbor_woodland', 'shrub_land', 'natural_grassland', 'artificial_grassland',
    'river', 'lake', 'pond'
]
GID_classes_w_concepts = [
    ['industrial_land', 'factory area', 'industrial park', 'workshop zone', 'warehouse area', 'industrial facility'],
    ['urban_residential', 'apartment area', 'high-density housing', 'urban housing', 'residential block', 'city community'],
    ['rural_residential', 'village housing', 'countryside settlement', 'rural homestead', 'low-rise rural house', 'scattered village'],
    ['traffic_land', 'road network', 'highway', 'urban road', 'transport corridor', 'parking and driveway'],
    ['paddy_field', 'rice field', 'paddy rice farmland', 'flooded field', 'irrigated paddy', 'cultivated paddy'],
    ['irrigated_land', 'irrigated farmland', 'canal-irrigated field', 'well-irrigated cropland', 'watered field', 'intensive irrigated land'],
    ['dry_cropland', 'rainfed cropland', 'dry farming field', 'non-irrigated farmland', 'upland crop field', 'seasonal cropland'],
    ['garden_plot', 'orchard', 'vegetable garden', 'economic crop field', 'plantation', 'horticultural land'],
    ['arbor_woodland', 'forest land', 'tree plantation', 'tall woodland', 'closed canopy forest', 'tree-covered area'],
    ['shrub_land', 'bushland', 'scrub vegetation', 'low shrub area', 'shrub-covered slope', 'sparse shrub'],
    ['natural_grassland', 'native grassland', 'pasture', 'meadow', 'natural herbaceous area', 'grazing land'],
    ['artificial_grassland', 'man-made grassland', 'artificial lawn', 'urban green grass', 'reclaimed grassland', 'managed grass field'],
    ['river', 'river channel', 'stream', 'watercourse', 'flowing water', 'river system'],
    ['lake', 'water lake', 'inland lake', 'large water body', 'open water surface', 'natural lake'],
    ['pond', 'fish pond', 'small water pond', 'artificial pond', 'storage pond', 'still water body'],
]

# =========================
# 5. MER / MSL
# =========================
MER_classes = ['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks', 'Tracks', 'Shadows', 'Unknown', 'Background']
MER_classes_w_concepts = [
    ['Martian Soil', 'mars soil', 'red soil', 'martian regolith', 'planetary soil', 'extraterrestrial soil'],
    ['Sands', 'martian sand', 'fine grains', 'loose sand', 'desert sand', 'silica sand'],
    ['Gravel', 'martian gravel', 'coarse gravel', 'pebbles', 'stone fragments', 'rocky gravel'],
    ['Bedrock', 'martian bedrock', 'solid rock', 'bedrock layer', 'parent rock', 'foundation rock'],
    ['Rocks', 'martian rocks', 'boulders', 'stone blocks', 'rock formations', 'large rocks'],
    ['Tracks', 'rover tracks', 'vehicle tracks', 'wheel tracks', 'trail marks', 'path imprints'],
    ['Shadows', 'martian shadows', 'rock shadows', 'shadow regions', 'dark areas', 'sun shadows'],
    ['Unknown', 'unidentified object', 'unknown material', 'ambiguous feature', 'unclassified', 'mysterious object'],
    ['Background', 'martian background', 'scene background', 'context area', 'surroundings', 'environment background']
]

def single_template(save_path, class_names, model, device):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings


def single_template_concept_avg(save_path, class_concept_list, model, device, dataset_name: str):
    all_concepts = [j for sub in class_concept_list for j in sub]
    print(f"Processing {len(all_concepts)} concepts for {dataset_name}...")

    with torch.no_grad():
        tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_concepts]).to(device)
        concept_embeddings = model.encode_text(tokens)
        print(f"Concept embeddings shape: {concept_embeddings.shape}")

        avg_concept_embeddings = []
        concept_idx = 0

        if dataset_name == "whdld":
            dataset_classes = WHDLD_classes
        elif dataset_name == "loveda":
            dataset_classes = LOVEDA_classes
        elif dataset_name == "potsdam":
            dataset_classes = Potsdam_classes
        elif dataset_name == "gid":
            dataset_classes = GID_classes
        elif dataset_name in ["mer", "msl"]: 
            dataset_classes = MER_classes
        else:
            raise ValueError(dataset_name)

        for i, concepts in enumerate(class_concept_list):
            n_concepts = len(concepts)
            print(f"Class {i} ({dataset_classes[i]}): {concepts}")
            class_concepts = concept_embeddings[concept_idx:concept_idx + n_concepts]
            avg_concept = torch.sum(class_concepts, dim=0) / n_concepts
            avg_concept_embeddings.append(avg_concept)
            concept_idx += n_concepts

        avg_concept_embeddings = torch.stack(avg_concept_embeddings, dim=0)
        avg_concept_embeddings = avg_concept_embeddings / avg_concept_embeddings.norm(dim=-1, keepdim=True)

        if save_path:
            np.save(save_path, avg_concept_embeddings.cpu().numpy())
    return avg_concept_embeddings


def aggregate_concept_predictions(pred, class_to_concept_idxs):
    B, _, H, W = pred.shape
    agg_pred = torch.zeros(B, len(class_to_concept_idxs), H, W, device=pred.device)
    for cls_i, conc_i in class_to_concept_idxs.items():
        agg_pred[:, cls_i] = pred[:, conc_i].max(dim=1).values
    return agg_pred


def flatten_class_concepts(class_concepts):
    concepts = []
    concept_to_class_idx = {}
    class_to_concept_idxs = {}
    for i, cls_concepts in enumerate(class_concepts):
        for concept in cls_concepts:
            concept_to_class_idx[len(concepts)] = i
            class_to_concept_idxs.setdefault(i, []).append(len(concepts))
            concepts.append(concept)
    return concepts, concept_to_class_idx, class_to_concept_idxs


def get_class_to_concept_idxs(dataset_name):
    if dataset_name == "whdld":
        c = WHDLD_classes_w_concepts
    elif dataset_name == "loveda":
        c = LOVEDA_classes_w_concepts
    elif dataset_name == "potsdam":
        c = Potsdam_classes_w_concepts
    elif dataset_name == "gid":
        c = GID_classes_w_concepts
    elif dataset_name in ["mer", "msl"]:
        c = MER_classes_w_concepts
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    _, _, class_to_concept_idxs = flatten_class_concepts(c)
    return class_to_concept_idxs


def prepare_text_embedding(save_path, dataset_name, model, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if dataset_name == "whdld":
        class_names = WHDLD_classes
        class_concepts = WHDLD_classes_w_concepts
        concept_num = 6
    elif dataset_name == "loveda":
        class_names = LOVEDA_classes
        class_concepts = LOVEDA_classes_w_concepts
        concept_num = 5
    elif dataset_name == "potsdam":
        class_names = Potsdam_classes
        class_concepts = Potsdam_classes_w_concepts
        concept_num = 5
    elif dataset_name == "gid":
        class_names = GID_classes
        class_concepts = GID_classes_w_concepts
        concept_num = 6
    elif dataset_name in ["mer", "msl"]:
        class_names = MER_classes
        class_concepts = MER_classes_w_concepts
        concept_num = 6
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    if save_path.endswith(f"{dataset_name}_single.npy"):
        print(f"Generating single template embeddings for {dataset_name}...")
        single_template(save_path, class_names, model, device)

    elif save_path.endswith(f"{dataset_name}_concept{concept_num}_single.npy"):
        print(f"Generating flat concept embeddings for {dataset_name}...")
        flat_concepts, _, _ = flatten_class_concepts(class_concepts)
        single_template(save_path, flat_concepts, model, device)

    elif save_path.endswith(f"{dataset_name}_conceptavg{concept_num}_single.npy"):
        print(f"Generating average concept embeddings for {dataset_name}...")
        single_template_concept_avg(save_path, class_concepts, model, device, dataset_name)

    else:
        print(f"Generating average concept embeddings for {dataset_name} (Fallback path match)...")
        single_template_concept_avg(save_path, class_concepts, model, device, dataset_name)

    print(f"{dataset_name} text embeddings saved to: {save_path}")


# =========================================================
# main
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare text embeddings for remote-sensing datasets")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    datasets = ['whdld', 'loveda', 'potsdam', 'gid', 'mer', 'msl']
    
    embedding_types = ['single', 'concept6_single', 'conceptavg6_single']

    output_dir = 'configs/_base_/datasets/text_embedding/'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    model, _ = clip.load('ViT-B/16', device)
    print(f"CLIP model loaded on {device}")

    for dataset in datasets:
        print(f"\n=== Processing {dataset.upper()} dataset ===")
        for embedding_type in embedding_types:
            etype = embedding_type

            if dataset in ['loveda', 'potsdam']:
                if 'concept6' in etype:
                    etype = etype.replace('concept6', 'concept5')
                if 'conceptavg6' in etype:
                    etype = etype.replace('conceptavg6', 'conceptavg5')

            save_path = f'{output_dir}/{dataset}_{etype}.npy'
            print(f"--- Generating {etype} for {dataset} ---")
            try:
                prepare_text_embedding(save_path, dataset, model, device)
            except Exception as e:
                print(f"Error generating {etype} for {dataset}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n=== All datasets and embedding types processed successfully ===")