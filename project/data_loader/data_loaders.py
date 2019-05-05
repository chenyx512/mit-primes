from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):

    def __init__(self, dataset, batch_size=32, shuffle=True, test_split=0,
                 num_workers=1, pin_memory=True):
        super().__init__(dataset, batch_size, shuffle, test_split, num_workers,
                         pin_memory=pin_memory)

