import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

def extract_embeddings(checkpoint_path, dataset_path, output_path, batch_size=32):
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if isinstance(ckpt, dict):
        from fairchem.core.common.utils import load_model_and_weights_from_checkpoint
        model = load_model_and_weights_from_checkpoint(checkpoint_path)
    else:
        # Handle MLIPInferenceCheckpoint
        from fairchem.core.units.mlip_unit.utils import load_inference_model
        model, _ = load_inference_model(checkpoint_path, return_checkpoint=True)
        
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading dataset from {dataset_path}...")
    # Initialize AseDBDataset with config, explicitly requesting custom info keys
    dataset_config = {
        "src": dataset_path,
        "a2g_args": {
            "r_custom_info_keys": ["atom_weight", "cluster_partition", "nearest_metal_idx", "cluster_center", "system_id"]
        }
    }
    dataset = AseDBDataset(dataset_config)
    print(f"Dataset size: {len(dataset)} structures.")

    all_embeddings = []
    all_tags = []
    all_atomic_numbers = []
    all_ids = []
    all_system_ids = []
    all_cluster_centers = []
    all_atom_weights = []
    all_nearest_metal_idx = []
    all_nearest_metal_mask = []

    print("Extracting embeddings...")
    data_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            data_list.append(data)
            
            # Process in batches
            if len(data_list) == batch_size or i == len(dataset) - 1:
                batch = atomicdata_list_to_batch(data_list).to(device)
                
                # UMA backbone expects charge, spin, and dataset
                if not hasattr(batch, 'charge') or getattr(batch, 'charge', None) is None:
                    batch.charge = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
                if not hasattr(batch, 'spin') or getattr(batch, 'spin', None) is None:
                    batch.spin = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
                if not hasattr(batch, 'dataset') or getattr(batch, 'dataset', None) is None:
                    # Find a valid dataset name from the model's dataset_mapping
                    default_ds_name = "data"
                    if hasattr(model, 'backbone') and hasattr(model.backbone, 'csd_embedding'):
                        try:
                            ds_mapping = model.backbone.csd_embedding.dataset_embedding.dataset_mapping
                            if ds_mapping:
                                default_ds_name = next(iter(ds_mapping.keys()))
                        except Exception:
                            pass
                    batch.dataset = [default_ds_name] * batch.num_graphs
                
                # Directly call backbone to bypass the output heads
                emb = model.backbone(batch)
                
                # UMA's node_embedding is [num_atoms, sph_feature_size, sphere_channels]
                # The l=0 (scalar) invariant embedding is at index 0 of sphere_channels
                node_features = emb["node_embedding"][:, 0, :]
                
                # Move to CPU and split back to individual structures
                node_features = node_features.cpu().numpy()
                natoms = batch.natoms.cpu().numpy()
                batch_ptr = np.concatenate([[0], np.cumsum(natoms)])
                
                tags = batch.tags.cpu().numpy() if hasattr(batch, 'tags') else None
                atomic_numbers = batch.atomic_numbers.cpu().numpy()
                ids = batch.id if hasattr(batch, 'id') else [str(j) for j in range(i - len(data_list) + 1, i + 1)]
                
                for j in range(len(data_list)):
                    start = batch_ptr[j]
                    end = batch_ptr[j + 1]
                    
                    struct_emb = node_features[start:end]
                    all_embeddings.append(struct_emb)
                    
                    if tags is not None:
                        all_tags.append(tags[start:end])
                    all_atomic_numbers.append(atomic_numbers[start:end])
                    
                    cur_id = ids[j] if isinstance(ids, list) else ids[j]
                    all_ids.append(cur_id)
                    
                    if hasattr(batch, 'system_id'):
                        sys_id = batch.system_id[j] if isinstance(batch.system_id, list) else batch.system_id[j]
                    else:
                        sys_id = ""
                    all_system_ids.append(sys_id)
                    
                    if hasattr(batch, 'cluster_center'):
                        cc_val = batch.cluster_center[j]
                        cc = cc_val.cpu().numpy() if torch.is_tensor(cc_val) else np.array(cc_val)
                    else:
                        cc = np.zeros(3)
                    all_cluster_centers.append(cc)
                    
                    if hasattr(batch, 'atom_weight'):
                        # atom_weight is a node attribute
                        aw_val = batch.atom_weight[start:end]
                        aw = aw_val.cpu().numpy() if torch.is_tensor(aw_val) else np.array(aw_val)
                    else:
                        aw = np.ones(end - start, dtype=float)
                    all_atom_weights.append(aw)
                    
                    if hasattr(batch, 'nearest_metal_idx'):
                        nmi_val = batch.nearest_metal_idx[j]
                        nmi = int(nmi_val.item()) if torch.is_tensor(nmi_val) else int(nmi_val)
                    else:
                        nmi = -1
                    all_nearest_metal_idx.append(nmi)
                    
                    # Convert the scalar index to a boolean mask per structure
                    mask = np.zeros(end - start, dtype=bool)
                    if 0 <= nmi < (end - start):
                        mask[nmi] = True
                    all_nearest_metal_mask.append(mask)
                
                # Clear batch
                data_list = []

    print(f"Saving extracted embeddings to {output_path}...")
    # We save it as an npz file which supports ragged arrays (lists of arrays with different sizes)
    save_dict = {
        "embeddings": np.array(all_embeddings, dtype=object),
        "atomic_numbers": np.array(all_atomic_numbers, dtype=object),
        "ids": np.array(all_ids, dtype=str),
        "system_id": np.array(all_system_ids, dtype=str),
        "cluster_center": np.array(all_cluster_centers),
        "atom_weight": np.array(all_atom_weights, dtype=object),
        "nearest_metal_idx": np.array(all_nearest_metal_idx),
        "nearest_metal_mask": np.array(all_nearest_metal_mask, dtype=object)
    }
    if len(all_tags) > 0:
        save_dict["tags"] = np.array(all_tags, dtype=object)
        
    np.savez(output_path, **save_dict)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract UMA Node Embeddings from a Dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the UMA model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the LMDB or ASE DB dataset")
    parser.add_argument("--output", type=str, default="uma_embeddings.npz", help="Output .npz file to save the embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    extract_embeddings(args.checkpoint, args.dataset, args.output, args.batch_size)
