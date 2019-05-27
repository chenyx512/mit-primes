import argparse

import cv2
import torch
from tqdm import tqdm
import numpy as np

from data_loader.data_loaders import DataLoader
import data_loader.event_fram_dataset as dataset_module
from parse_config import ConfigParser
from model.model import Model
from utils.CSVDict import CSVDict


def display(out, video_frame, event_frame, truth, prediction):
    """ This function displays the event frame and the video frame,
    also visualizes the steering angle prediction / truth """

    video_frame = cv2.resize(video_frame, (240, 180))
    # event frame from torch 1*2*180*240 to numpy 180*240*2
    event_frame = event_frame.numpy().squeeze(0).transpose(1, 2, 0)
    # change to bgr, with B zeros, R/G showing the addition/reduction
    event_frame = np.pad(event_frame, [(0, 0), (0, 0), (1, 0)],
                         mode='constant', constant_values=0)
    event_frame *= 255 * 10
    event_frame = np.clip(event_frame, 0, 255)
    frame = np.concatenate((video_frame, event_frame), axis=1)
    frame = np.pad(frame, [(20, 20), (0, 0), (0, 0)], mode='constant',
                   constant_values=255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'truth {truth.item():3.1f}'+
                f' predict {prediction.item():3.1f}',
                (0, 10), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA, False)

    cv2.line(frame, (240, 200), (240, 220), (0, 0, 0), 1)
    cv2.circle(frame, (240 - truth, 205), 3, (0, 255, 0), -1)
    cv2.circle(frame, (240 - prediction, 215), 3, (0, 0, 255), -1)
    frame = frame.astype(np.uint8)
    out.write(frame)


def main(config: ConfigParser):
    dataset = config.initialize('dataset', dataset_module)
    data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=8,
                             pin_memory=False)

    video = cv2.VideoCapture('data/camera_front.avi')
    current_index = 0
    ret, video_frame = video.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc,
                          1/dataset.integration_time, (480, 220))

    state = torch.load(config.resume)

    model = Model()
    model.load_state_dict(state['model_state'])
    model.eval()

    time2index = CSVDict('data/camera_front.csv', key_index=1, value_index=0)

    with torch.no_grad():
        for i, (event_frame, truth) in enumerate(tqdm(data_loader)):
            time = dataset.frame_time[i]
            frame_index = int(time2index[time])
            while current_index < frame_index:
                ret, video_frame = video.read()
                current_index += 1
                if not ret:
                    raise Exception('video ends')

            truth *= dataset.steering_angle_factor
            prediction = model(event_frame) * dataset.steering_angle_factor
            display(out, video_frame, event_frame, truth, prediction)
            if i == 5000:
                break

    video.release()
    out.release()


args = argparse.ArgumentParser()
args.add_argument('-r', '--resume',
                  default='saved/models/test0522_best/best.pth',
                  type=str, help='path to latest checkpoint (default: None)')

config_parser = ConfigParser(args, [])
main(config_parser)
