from torch.utils.data.sampler import SubsetRandomSampler

from base.base_data_loader import BaseDataLoader
from  utils.CSVDict import CSVDict

class DataLoader(BaseDataLoader):
    """The main data loader for training.

    The dataset is first cut to include only periods when the car is driving
    faster than a given speed. Then it is divided into consecutive periods of
    40 sec for training and 20 sec for testing

    Args:
        dataset (Dataset): the dataset to be loaded from
        speed_path (string): the path to the time2speed csv file
        min_speed (float): the minimum speed of the car in training/test set
        batch_size in m/s
        num_workers (int, optional): number of processes to use (default: 1)
        pin_memory (bool, optional): whether to copy tensors into CUDA pinned
            memory before returning them. (default: True)
        drop_last (bool, optional): whether to drop the last incomplete batch.
            (default: True)
    """

    def __init__(self, dataset, speed_path, min_speed, batch_size,
                 num_workers=1, pin_memory=True, drop_last=True):
        self.time2speed = CSVDict(speed_path)
        self.min_speed = min_speed

        super().__init__(dataset, batch_size, False, 0, num_workers,
                         pin_memory=pin_memory, drop_last=drop_last)

    def _split_sampler(self, split):
        """ Note: the arguments are inherited and have no actual effects """

        self.train_indexes = []
        self.test_indexes = []

        fps = int(1 / self.dataset.integration_time)
        cnt = 0

        for i in range(self.length):
            if self.time2speed[self.dataset.frame_time[i]] < self.min_speed:
                continue

            if cnt < 40 * fps:
                self.train_indexes.append(i)
            else:
                self.test_indexes.append(i)

            cnt += 1
            if cnt == 60 * fps:
                cnt = 0

        return SubsetRandomSampler(self.train_indexes), \
            SubsetRandomSampler(self.test_indexes)
