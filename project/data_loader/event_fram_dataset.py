from bisect import bisect
import logging

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from utils.CSVDict import CSVDict


class EventFrameDataset(Dataset):
    """Dataset of event frames and corresponding steering angles

    This dataset accepts the event and steering angle data.
    It prepossesses the range of events for each frame of a given integration
    times, and the event frames are then integrated on the fly using the
    preprocessed range.

    Args:
        event_dir (str): the path of the event h5 file
        steering_angle_dir (str): the path of the steering angle csv file
        integration_time (float): the time span of each event frame, in
            seconds.
        max_pixel_value (int, optional): If it is None, the program will
            calculate this value. Specifying this value will save time. For MIT
            dataset, it is 17 for an integration time of 50ms. (default: None)
        num_process (int, optional): If max_pixel_value is None, this specifies
            the number of processes used to calculate max_pixel_value.
            (default: 8)
    """

    def __init__(self, event_dir, steering_angle_dir, integration_time,
                 max_pixel_value=None, num_process=8):
        self.logger = logging.getLogger('dataset')

        datafile = h5py.File(event_dir, 'r')
        event_time = np.array(datafile['time'])
        # the steering angles are clamped and normalize by 3 times their
        # standard deviation
        steering_angle = CSVDict(steering_angle_dir, norm_factor=3,
                                 clamp_factor=3)
        self.steering_angle_factor = steering_angle.std * 3
        self.logger.info(f'steering_angle normalized by \
            {self.steering_angle_factor}')

        self.event_x_pos = np.array(datafile['x_pos'])
        self.event_y_pos = np.array(datafile['y_pos'])
        self.event_polarity = np.array(datafile['polarity'])

        self.integration_time = integration_time
        self.tot_frame = int((event_time[-1] - event_time[0]) /
                             integration_time)
        self.frame_events_range = []
        self.frame_steering_angle = []
        self.frame_time = []

        # calculate the range of events of each event frame
        left_event_index = 0
        for frame_index in range(self.tot_frame):
            self.frame_time.append(event_time[0] + \
                (frame_index + 0.5) * integration_time)
            right_event_index = bisect(event_time, self.frame_time[frame_index])
            self.frame_events_range.append(range(left_event_index,
                                                 right_event_index))
            self.frame_steering_angle.append(
                torch.tensor([steering_angle[self.frame_time[-1]]]))
            left_event_index = right_event_index

        self.max_pixel_value = max_pixel_value
        if not self.max_pixel_value:
            self.logger.info('start calculating max_pixel_value')
            self._calculate_max_pixel_value(num_process)
        logging.info(f'Event Frame normalized by {self.max_pixel_value}')

    def _calculate_max_pixel_value(self, num_process):
        """use multiprocessing to speed up the calculation of max_pixel_value
        for normalization"""
        pipes = []
        for i in range(num_process):
            recv_end, send_end = mp.Pipe(False)
            pipes.append(recv_end)
            p = mp.Process(target=self._update_max,
                           args=(range(int(i * self.tot_frame / num_process),
                                 int((i + 1) * self.tot_frame / num_process)),
                                 send_end))
            p.daemon = True
            p.start()
        self.max_pixel_value = max(conn.recv() for conn in pipes)

    def _update_max(self, frame_range, conn):
        max_pixel_value = max(self[frame_index][0].max().item()
                              for frame_index in frame_range)
        conn.send(max_pixel_value)

    def __len__(self):
        return self.tot_frame

    def __getitem__(self, frame_index):
        frame = torch.zeros(2, 180, 240)
        for i in self.frame_events_range[frame_index]:
            frame[0 if self.event_polarity[i] else 1][self.event_y_pos[i]] \
                [self.event_x_pos[i]] += 1
        # normalize to [0, 1] using max_pixel_value if it exists
        if self.max_pixel_value:
            frame /= self.max_pixel_value
        return frame, self.frame_steering_angle[frame_index]
