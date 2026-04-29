import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
    WeightedRandomSampler,
    random_split,
)
from tqdm.auto import tqdm

from plot2eq.data.augmentation import HandDrawnAugmentation
from plot2eq.data.dataset import SymbolicDataset


class AugmentedSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        points, tokens = self.base_dataset[self.indices[idx]]
        if self.transform:
            points = self.transform(points)
        return points, tokens


def build_dataloaders(
    data_dir: str, batch_size: int, val_split: float = 0.05, seed: int = 42
):

    print("Loading base dataset...")
    base_dataset = SymbolicDataset(
        data_dir, drawn_augmentation=False, map_location="cpu"
    )

    dataset_size = len(base_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(dataset_size),  # type: ignore
        [train_size, val_size],
        generator=generator,  # type: ignore
    )

    token_lengths = (
        base_dataset.tokens != base_dataset.tokenizer.token_map["<pad>"]
    ).sum(axis=1)  # type: ignore

    train_token_lengths = token_lengths[train_indices.indices]
    weights = 1.0 / (train_token_lengths.float() + 10.0)

    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_indices), replacement=True
    )

    train_transform = HandDrawnAugmentation(p=0.8)
    train_dataset = AugmentedSubset(
        base_dataset, train_indices, transform=train_transform
    )

    print("Pre-computing deterministic noisy validation set...")
    val_transform = HandDrawnAugmentation(p=1.0)

    torch.manual_seed(1337)

    val_points_list, val_tokens_list = [], []
    for idx in tqdm(val_indices, desc="Caching Val Noise", leave=False):  # type: ignore
        clean_points, tokens = base_dataset[idx]
        noisy_points = val_transform(clean_points)

        val_points_list.append(noisy_points)
        val_tokens_list.append(tokens)

    val_dataset = TensorDataset(
        torch.stack(val_points_list), torch.stack(val_tokens_list)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        len(base_dataset.tokenizer.tokens),
        base_dataset.tokenizer.token_map["<pad>"],
    )
