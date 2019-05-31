from torch.utils.data.sampler import SubsetRandomSampler

from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    """The main data loader for training.

    The dataset is divided into consecutive periods of 40 sec for training and
    20 sec for testing

    Args:
        dataset (Dataset): the dataset to be loaded from
        batch_size (int)
        num_workers (int, optional): number of processes to use (default: 1)
        pin_memory (bool, optional): whether to copy tensors into CUDA pinned
            memory before returning them. (default: True)
        drop_last (bool, optional): whether to drop the last incomplete batch.
            (default: True)
    """

    def __init__(self, dataset, batch_size, num_workers=1, pin_memory=True,
                 drop_last=True):
        super().__init__(dataset, batch_size, False, 0, num_workers,
                         pin_memory=pin_memory, drop_last=drop_last)

    def _split_sampler(self, split):
        """ Note: the arguments are inherited and have no actual effects """

        train_indexes = []
        test_indexes = []

        fps = int(1 / self.dataset.integration_time)

        for i in range(0, self.length, 60 * fps):
            train_indexes += list(range(i, min(self.length, i + 40 * fps)))
        for i in range(40 * fps, self.length, 60 * fps):
            test_indexes += list(range(i, min(self.length, i + 20 * fps)))

        return SubsetRandomSampler(train_indexes), \
            SubsetRandomSampler(test_indexes)
