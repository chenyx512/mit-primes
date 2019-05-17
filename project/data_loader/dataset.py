from bisect import bisect
import logging

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from data_loader.CSVDict import CSVDict


class EventFrameDataset(Dataset):
    """Dataset of event frames and corresponding steering angles

    This dataset accepts the event and steering angle data.
    It prepossesses the range of events for each frame of a given integration times,
    and the event frames are then integrated on the fly using the preprocessed range.
    """

    def __init__(self, event_dir, steering_angle_dir, integration_time,
                 partial_range=None, num_process=8, max_pixel_value=None):
        print('start initiating dataset')
        file = h5py.File(event_dir, 'r')
        event_time = np.array(file['time'])
        steering_angle = CSVDict(steering_angle_dir, is_norm=True, clamp=3)
        logging.info(f'steering_angle normalized by {steering_angle.std}')
        self.steering_angle_std = steering_angle.std

        self.event_x_pos = np.array(file['x_pos'])
        self.event_y_pos = np.array(file['y_pos'])
        self.event_polarity = np.array(file['polarity'])

        self.tot_frame = int((event_time[partial_range - 1 if partial_range else -1]
                              - event_time[0]) / integration_time)
        self.frame_events_range = []
        self.frame_steering_angle = []

        # calculate the range of events of each event frame
        left_event_index = 0
        for frame_index in range(self.tot_frame):
            frame_time = event_time[0] + (frame_index + 0.5) * integration_time
            right_event_index = bisect(event_time, frame_time)
            self.frame_events_range.append(range(left_event_index, right_event_index))
            self.frame_steering_angle.append(torch.tensor([steering_angle[frame_time]]))
            left_event_index = right_event_index

        self.max_pixel_value = max_pixel_value
        if not self.max_pixel_value:
            # use multiprocessing to speed up the calculation of max_pixel_value for normalization
            # question: is it a good idea to normalize using max, or standard deviation ???
            pipes = []
            for i in range(num_process):
                recv_end, send_end = mp.Pipe(False)
                pipes.append(recv_end)
                p = mp.Process(target=self.update_max,
                               args=(range(int(i * self.tot_frame / num_process),
                                           int((i + 1) * self.tot_frame / num_process)),
                                     send_end))
                p.daemon = True
                p.start()
            self.max_pixel_value = max(conn.recv() for conn in pipes)
        elif partial_range:
            logging.warn('both partial_range and manual max_pixel value are enabled')
        logging.info(f'Event Frame normalized by {self.max_pixel_value}')

    def update_max(self, frame_range, conn):
        max_pixel_value = max(self[frame_index][0].max().item() for frame_index in frame_range)
        conn.send(max_pixel_value)

    def __len__(self):
        return self.tot_frame

    def __getitem__(self, frame_index):
        frame = torch.zeros(2, 180, 240)
        for i in self.frame_events_range[frame_index]:
            frame[0 if self.event_polarity[i] else 1][self.event_y_pos[i]] \
                [self.event_x_pos[i]] += 1
        # normalize to [0, 1]
        if self.max_pixel_value:
            frame /= self.max_pixel_value
        return frame, self.frame_steering_angle[frame_index]
