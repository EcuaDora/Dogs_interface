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
from utilites import *



data_path = os.path.join(r'C:\Users\vladm\OneDrive\Документы\GitHub\GitHub\Dogs_interface\тестовые данные')

group_names = ['caff_50mg', 'caff_100mg', 'caff_control', '9j_1mg', '9j_100mg', '9j_control']

target_path = os.path.join(r'C:\Users\vladm\OneDrive\Документы\GitHub\GitHub\Dogs_interface\StatTools\result', )

os.makedirs(target_path, exist_ok=True)

conventional_analysis(data_path, group_names, target_path)