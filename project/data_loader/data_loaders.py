from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    """The main data loader for training

    Args:
        dataset (Dataset): the dataset to be loaded from
        batch_size (int)
        shuffle (bool, optional): whether to shuffle data at every epoch
            (default: True)
        test_split (optional): an integer indicating length of test_data,
                    or a float indicating ratio of test_data in total data.
                    (default: 0)
        num_workers (int, optional): number of processes to use (default: 1)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch. (default: default_collate)
        pin_memory (bool, optional): whether to copy tensors into CUDA pinned
            memory before returning them. (default: True)

    Note: if test_split is not 0, shuffle will always be performed on
        train_loader
    """

    def __init__(self, dataset, batch_size, shuffle=True, test_split=0,
                 num_workers=1, pin_memory=True):
        super().__init__(dataset, batch_size, shuffle, test_split, num_workers,
                         pin_memory=pin_memory)
