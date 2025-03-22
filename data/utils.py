import torch

def collate_fn(batch_data):
    return {
        "input": torch.vstack([b["input"] for b in batch_data]),
        "target": torch.vstack([b["target"] for b in batch_data]),
        "mask": torch.vstack([b["mask"] for b in batch_data]),
    }