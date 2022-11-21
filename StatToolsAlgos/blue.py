import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pandas as pd
import os
import glob
import math
import seaborn as sns
import matplotlib.pyplot as plt
import StatTools
import sys
from utilites import *


data_path = os.path.join('data', 'original', 'trajectories')
target_path = os.path.join('data', 'original', 'parameters')
os.makedirs(target_path, exist_ok=True)



drugs_data = {
    '9j': {
        'control' : ['data/original/trajectories/9j_control_3.csv', 'data/original/trajectories/9j_control_8.csv', 'data/original/trajectories/9j_control_11.csv', 'data/original/trajectories/9j_control_12.csv', 'data/original/trajectories/9j_control_4.csv', 'data/original/trajectories/9j_control_7.csv', 'data/original/trajectories/9j_control_5.csv', 'data/original/trajectories/9j_control_10.csv', 'data/original/trajectories/9j_control_2.csv', 'data/original/trajectories/9j_control_0.csv'],

        '1mg': ['data/original/trajectories/9j_1mg_9.csv', 'data/original/trajectories/9j_1mg_14.csv', 'data/original/trajectories/9j_1mg_15.csv', 'data/original/trajectories/9j_1mg_10.csv', 'data/original/trajectories/9j_1mg_4.csv', 'data/original/trajectories/9j_1mg_5.csv', 'data/original/trajectories/9j_1mg_3.csv', 'data/original/trajectories/9j_1mg_8.csv', 'data/original/trajectories/9j_1mg_1.csv', 'data/original/trajectories/9j_1mg_11.csv', 'data/original/trajectories/9j_1mg_2.csv', 'data/original/trajectories/9j_1mg_7.csv', 'data/original/trajectories/9j_1mg_13.csv', 'data/original/trajectories/9j_1mg_6.csv', 'data/original/trajectories/9j_1mg_0.csv', 'data/original/trajectories/9j_1mg_12.csv'],

        '100mg':  ['data/original/trajectories/9j_100mg_12.csv', 'data/original/trajectories/9j_100mg_14.csv', 'data/original/trajectories/9j_100mg_2.csv', 'data/original/trajectories/9j_100mg_8.csv', 'data/original/trajectories/9j_100mg_0.csv', 'data/original/trajectories/9j_100mg_6.csv', 'data/original/trajectories/9j_100mg_3.csv', 'data/original/trajectories/9j_100mg_9.csv', 'data/original/trajectories/9j_100mg_13.csv', 'data/original/trajectories/9j_100mg_11.csv', 'data/original/trajectories/9j_100mg_10.csv', 'data/original/trajectories/9j_100mg_5.csv', 'data/original/trajectories/9j_100mg_7.csv', 'data/original/trajectories/9j_100mg_1.csv', 'data/original/trajectories/9j_100mg_4.csv'],
    },

    'caff': {
        'control': ['data/original/trajectories/caff_control_11.csv', 'data/original/trajectories/caff_control_0.csv', 'data/original/trajectories/caff_control_1.csv', 'data/original/trajectories/caff_control_10.csv', 'data/original/trajectories/caff_control_7.csv', 'data/original/trajectories/caff_control_8.csv', 'data/original/trajectories/caff_control_9.csv', 'data/original/trajectories/caff_control_5.csv', 'data/original/trajectories/caff_control_12.csv', 'data/original/trajectories/caff_control_2.csv', 'data/original/trajectories/caff_control_6.csv', 'data/original/trajectories/caff_control_3.csv', 'data/original/trajectories/caff_control_4.csv', 'data/original/trajectories/caff_control_13.csv'],

        '50mg': ['data/original/trajectories/caff_50mg_6.csv', 'data/original/trajectories/caff_50mg_0.csv', 'data/original/trajectories/caff_50mg_7.csv', 'data/original/trajectories/caff_50mg_12.csv', 'data/original/trajectories/caff_50mg_1.csv', 'data/original/trajectories/caff_50mg_11.csv', 'data/original/trajectories/caff_50mg_10.csv', 'data/original/trajectories/caff_50mg_5.csv', 'data/original/trajectories/caff_50mg_2.csv', 'data/original/trajectories/caff_50mg_3.csv', 'data/original/trajectories/caff_50mg_4.csv', 'data/original/trajectories/caff_50mg_9.csv'],

        '100mg': ['data/original/trajectories/caff_100mg_10.csv', 'data/original/trajectories/caff_100mg_7.csv', 'data/original/trajectories/caff_100mg_11.csv', 'data/original/trajectories/caff_100mg_5.csv', 'data/original/trajectories/caff_100mg_3.csv', 'data/original/trajectories/caff_100mg_9.csv', 'data/original/trajectories/caff_100mg_14.csv', 'data/original/trajectories/caff_100mg_8.csv', 'data/original/trajectories/caff_100mg_0.csv', 'data/original/trajectories/caff_100mg_2.csv', 'data/original/trajectories/caff_100mg_13.csv', 'data/original/trajectories/caff_100mg_1.csv', 'data/original/trajectories/caff_100mg_12.csv', 'data/original/trajectories/caff_100mg_4.csv', 'data/original/trajectories/caff_100mg_6.csv'],

    }
}







conventional_analysis(drugs_data,  target_path)





