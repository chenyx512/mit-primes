import unittest

import torch

from data_loader.data_loaders import DataLoader
from data_loader.event_frame_dataset import EventFrameDataset


class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = EventFrameDataset('../data/events.h5', '../data/steering_can.csv', 0.05)

    def test_split(self):
        train_loader = DataLoader(self.dataset, test_split=100, batch_size=10, num_workers=10)
        test_loader = train_loader.get_test_loader()
        train_batches = sum(1 for _ in train_loader)
        test_batches = sum(1 for _ in test_loader)
        self.assertEqual(test_batches, 10)
        correct_train_batches = int((len(self.dataset) - 100 + 9) / 10)
        self.assertEqual(train_batches, correct_train_batches)

    def test_data(self):
        loader = DataLoader(self.dataset, num_workers=10)
        targets = torch.tensor([])
        eps = 1e-7
        for input, target in loader:
            min_input = input.min().item()
            self.assertGreaterEqual(min_input + eps, 0)
            max_input = input.max().item()
            self.assertGreaterEqual(1 + eps, max_input)

            self.assertNotAlmostEqual(input.std().item(), 0)

            min_target = target.min().item()
            self.assertGreaterEqual(min_target + eps, -3)
            max_target = target.max().item()
            self.assertGreaterEqual(3 + eps, max_target)

            targets = torch.cat((targets, target))

        target_std = targets.std().item()
        print(f'target std: {target_std}')
        self.assertNotAlmostEqual(target_std, 0)


if __name__ == '__main__':
    unittest.main()
