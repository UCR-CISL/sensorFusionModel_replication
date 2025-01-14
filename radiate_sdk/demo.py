import os
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate

from dataset_loader import RadiateDataset

# Configuration constants
DATASET_PATH = '/media/sudhanshu/Janice/sudhanshu/radiate'
BATCH_SIZE = 1
TRAIN_SPLIT_RATIO = 0.7

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors.
    :param batch: List of samples from the dataset.
    :return: Collated batch with lists for variable-sized data.
    """
    collated_batch = {}
    for key in batch[0].keys():
        # If data has variable sizes (e.g., bboxes), keep them as lists
        if key == 'bboxes' or key == 'classes':
            collated_batch[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors with consistent shapes
            collated_batch[key] = default_collate([sample[key] for sample in batch])
    return collated_batch

def prepare_datasets(dataset_path, train_split_ratio):
    """
    Prepare train and test datasets by loading sequences and splitting.
    :param dataset_path: Path to the dataset folder.
    :param train_split_ratio: Ratio of train dataset size to the total dataset size.
    :return: Combined train and test datasets.
    """
    # Get list of sequence folder names
    sequence_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

    all_train_datasets = []
    all_test_datasets = []

    for sequence_name in sequence_names:
        # Initialize dataset for the current sequence
        radiate_dataset = RadiateDataset(dataset_path, sequence_name)

        # Split into train and test datasets
        train_size = int(train_split_ratio * len(radiate_dataset))
        test_size = len(radiate_dataset) - train_size
        train_dataset, test_dataset = random_split(radiate_dataset, [train_size, test_size])

        # Add to combined datasets
        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)

    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_test_dataset = ConcatDataset(all_test_datasets)

    return combined_train_dataset, combined_test_dataset

def get_data_loaders():
    """
    Returns train and test DataLoaders.
    """
    combined_train_dataset, combined_test_dataset = prepare_datasets(DATASET_PATH, TRAIN_SPLIT_RATIO)
    train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(combined_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, test_loader

# def main():
#     """
#     Main function to prepare datasets and display data.
#     """
#     train_loader, test_loader = get_data_loaders()

#     # Display one batch for verification
#     for batch in train_loader:
#         print(f"Batch keys: {batch.keys()}")
#         print(batch['radar'].shape)
#         print(batch['lidar'].shape)
#         print(batch['camera_left'].shape)
#         print(batch['camera_right'].shape)
#         print(batch['bboxes'])
#         print(batch['classes'])
#         break

# if __name__ == "__main__":
#     main()