from typing import List, Any, Dict
import torch


def collate_batch(features: List[Any]) -> Dict[str, torch.Tensor]:
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if hasattr(first, "label") and first.label is not None:
        if type(first.label) is int:
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        batch = {"labels": labels}
    elif hasattr(first, "label_ids") and first.label_ids is not None:
        if type(first.label_ids[0]) is int:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
    return batch