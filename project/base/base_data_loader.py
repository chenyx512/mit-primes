import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """Base data loader that includes the splitting of test loader"""

    def __init__(self, dataset, batch_size, shuffle, test_split, num_workers,
                 collate_fn=default_collate, pin_memory=False):
        self.test_split = test_split
        self.shuffle = shuffle
        self.length = len(dataset)
        self.sampler, self.test_sampler = self._split_sampler(test_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': pin_memory
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0:
            return None, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.length, "test set size is larger than entire dataset."
            test_length = split
        else:
            test_length = int(self.length * split)

        indexes = np.arange(self.length)
        np.random.seed(0)
        np.random.shuffle(indexes)

        test_indexes = indexes[0:test_length]
        train_indexes = np.delete(indexes, np.arange(test_length))

        self.shuffle = False
        self.length = len(train_indexes)

        return SubsetRandomSampler(train_indexes), SubsetRandomSampler(test_indexes)

    def get_test_loader(self):
        if self.test_sampler is None:
            return None
        return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
