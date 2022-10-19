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
from StatTools.analysis.dpcca import dpcca
from StatTools.analysis.movmean import movmean
from StatTools.generators.base_filter import Filter


def get_object(frame: np.array, background, coords: list, filtering: bool = True) -> tuple:
    """
    Function for coordinates extraction from frame

    Args:
        frame (np.array): frame from the video
        background(np.array): frame without object 
        coords(list): list of limits for object finding [x_min, x_max, y_min, y_max]
        filtering (bool, optional): parameter for selecting the need for filtering; default = True

    Returns:
         coordinates(tuple): object coordinates: (x, y) or (None, None) if object not found 
                            or (None, None) if center of the contour is 0
    """

    x_min = coords[0]
    x_max = coords[1]
    y_min = coords[2]
    y_max = coords[3]

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    _, th1 = cv2.threshold(grayscale_img, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    _, th2 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [grayscale_img, 0, th1,
              blur, 0, th2]

    if filtering:
        binarized_image = images[1 * 3 + 2]
    else:
        binarized_image = images[0 * 3 + 2]

    bgd_substracted = cv2.absdiff(background, binarized_image)
    bgd_substracted = cv2.bitwise_not(bgd_substracted)
    bgd_substracted = cv2.GaussianBlur(bgd_substracted, (5, 5), 0)
    bgd_substracted = cv2.bitwise_not(bgd_substracted)
    (contours, hierarchy) = cv2.findContours(
        bgd_substracted.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scores = []
    crops = []
    img_contours = []

    for c in contours:
        scale = cv2.contourArea(c)

        if scale < 9 or scale > 1000:
            continue

        # get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(c)

        if x <= x_min or x + w >= x_max or y >= y_max or y <= y_min:
            continue

        crop_img = bgd_substracted[y: y + h, x: x + w]
        crops.append(crop_img)
        nonzero = crop_img[np.where(crop_img != 0)]
        scores.append(np.mean(nonzero))
        img_contours.append(c)

        bgd_substracted = cv2.rectangle(
            bgd_substracted, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if not scores:
        return (None, None)

    idx = np.argmax(scores)
    contour = img_contours[idx]

    # compute the center of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:
        print('Error: center of the contour = 0')
        return (None, None)

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    coordinates = (x, y)
    return coordinates


def cv_trajectory_extraction(video: str, coords_bordes: list, filtering_frame: bool = True) -> tuple:
    """
        Function for coordinates extraction from the video
        Args:
            video(str): path to video for processing
            coords_bordes(list): list of limits for object finding [x_min, x_max, y_min, y_max]
            filtering_frame (bool, optional): parameter for selecting the need for frame filtering; default = True
        Returns:
            coords_df(.csv file): file with coordinates
            len_count(int): number of frames
            none_count(int): number of gaps
    """

    if video == None:
        print('Video is None')

    elif None in coords_bordes:
        print('None is in coords_bordes')

    else:
        none_count = 0
        len_count = 0
        data = []
        clip = VideoFileClip(video)

        for i, frame in enumerate(clip.iter_frames()):
            if i == 0:
                grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, th1 = cv2.threshold(grayscale_img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                blur = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
                _, th2 = cv2.threshold(
                    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                images = [grayscale_img, 0, th1,
                          blur, 0, th2]
                if filtering_frame:
                    binarized_image = images[1 * 3 + 2]
                else:
                    binarized_image = images[0 * 3 + 2]
                background = binarized_image

            x, y = get_object(frame, background,
                              coords_bordes, filtering_frame)
            len_count += 1
            if x == None:
                x = np.NaN
                y = np.NaN
                none_count += 1
            data.append([x, y])

        # csv creating
        coords_df = pd.DataFrame(data, columns=['x', 'y'])
        coords_df = coords_df.round(1)
        coords_df.y = coords_df.y.interpolate()

    return (coords_df, len_count, none_count)


def get_pharm_statistics(data_file: str, min_len: int, mode: str, fps: float = 25) -> list:
    """
    Function for conventional analysis based on the estimation of multiple scalar metrics 

    Args:
        data_file(str): path to file for processing
        min_len(int): minimum trajectory length
        mode(str): 'model' or 'orig' trajectories, important for 'stop_duration' threshold
        fps(float): fps of video
        
    Returns:
        data_list(list): list with parametrs in that order: [name, total distance, average speed, max speed, stop duration, sum top time, crosses count, first ascent latency]
    """

    if mode == 'orig':
        threshold_caff = 0.12658227848101222 #threshold for stop duration - quantile (0.27) of speeds in caff control group 
        threshold_9j = 0.13448776627963102 #threshold for stop duration - quantile (0.27) of speeds in 9j control group 
    elif mode == 'model':
        threshold_caff = 0.43301794443107317
        threshold_9j = 0.8079598948655203
    data = pd.read_csv(data_file, index_col=0)
    data = data.dropna(axis=0, how='all')
    data.index = range(0, len(data))

    data_size = data.shape[0]  # Clipping trajectories to the minimum length
    drop_size = data_size - min_len
    data = data.drop(list(range(data_size-drop_size, data_size)))

    data_list = []
    name = os.path.splitext(os.path.basename(data_file))[0]
    data_list.append(name)

    times = data.index
    times = times - np.min(times)
    times = times * 1 / fps
    data['time'] = times

    x_min = np.min(data['x'])
    x_max = np.max(data['x'])
    y_max = np.max(data['y'])
    y_min = np.min(data['y'])

    data['x'] = (data['x'] - x_min) * 2 / \
        (x_max-x_min) - 1  # Data normalization
    data['y'] = (data['y'] - y_min) * 2 / (y_max-y_min) - 1
    s = np.stack([data['x'], data['y']], axis=1)

    # total distance
    length = np.sqrt(np.sum(np.diff(s, axis=0) ** 2, axis=1))
    dist = np.sum(length)
    data_list.append(dist)

    # average speed
    a_speed = dist / max(times)
    data_list.append(a_speed)

    # max speed
    N = 5
    prev = 0
    res = []
    time_diff = times[N] - times[0]

    for k in range(N, len(times[N:]), N):
        points = s[prev:k]
        dist1 = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        dist1 = np.sum(dist1)
        res.append(dist1 / time_diff)
        prev = k
    speeds = np.array(res)
    max_speed = np.max(speeds)
    data_list.append(max_speed)

    # stop duration
    if 'caff' in name:
        threshold = threshold_caff
    if '9j' in name:
        threshold = threshold_9j
    a = []
    for i in range(len(res)):
        if res[i] <= threshold:
            a.append(res[i])
    stop_dur = len(a) * time_diff
    data_list.append(stop_dur)

    # Sum top time
    cnt = 0
    for el in data['y']:
        if el <= 0:
            cnt += 1
    sum_time = cnt * 1 / 25
    data_list.append(sum_time)

    # Crosses count
    pos = (data['y'].values - 0) > 0
    npos = ~pos
    ind = (npos[:-1] & pos[1:]).nonzero()[0]
    crosses_count = len(ind)
    data_list.append(crosses_count)

    # First ascent latency
    if crosses_count == 0:
        data_list.append(None)
    else:
        first_time = data.loc[ind[0]].time
        data_list.append(first_time)

    return data_list


def tukeys_method(df: pd.DataFrame, variable: str, c: float = 0.25) -> list:
    """
        Function for finding outliers in dataframe

        Args:
            df(DataFrame): dataframe
            variable(str): parameter of interest
            c(float): quantile

        Returns:
            outliers(list): the list of outliers
    """
    q1 = df[variable].quantile(c)
    q3 = df[variable].quantile(1-c)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    # inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence

    outliers_prob = []
    outliers = []
    for x in df[variable]:
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(x)
    for x in df[variable]:
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers.append(round(x, 3))
    return outliers


def conventional_analysis(data_path: str, group_names: list, target_path: str, mode: str = 'orig'):
    """
    Function for conventional analysis based on the estimation of multiple scalar metrics. 
    Saves csv file with patameters for groups to target_path.

    Args:
        data_path(str): path to folder with trajectories
        group_names(list): list with experimental groups
        target_path(str): path for saving files
        mode(str): 'model' or 'orig' trajectories, important for 'stop_duration' threshold
    """
    params = ['total_distance', 'average_speed', 'max_speed', 'stop_duration',
              'sum_top_time', 'crosses_count', 'first_ascent_latency']
    min_len = None
    for csv_path in glob.glob(os.path.join(data_path, '*.csv')):
        df = pd.read_csv(csv_path, index_col=0)
        df = df.dropna(axis=0, how='all')
        if min_len == None:
            min_len = df.shape[0]
        elif df.shape[0] < min_len:
            min_len = df.shape[0]

    for group_name in group_names:
        group_data_list = []
        for file_path in tqdm(glob.glob(os.path.join(data_path, group_name + '*.csv'))):
            group_data_list.append(get_pharm_statistics(file_path, min_len, mode))

        group_data_df = pd.DataFrame(group_data_list, columns=['video_name', 'total_distance', 'average_speed', 'max_speed', 'stop_duration', 'sum_top_time',
                                                               'crosses_count', 'first_ascent_latency'])

        # Outliers removing from DataFrame.
        # Outliers replaced to np.nan.
        for param in params:
            outliers_list = tukeys_method(group_data_df, param)
            for row_index in range(group_data_df.shape[0]):
                if group_data_df.loc[row_index, param] in outliers_list:
                    group_data_df.loc[row_index, param] = np.nan
        csv_path = os.path.join(target_path, group_name + '_parameters.csv')
        group_data_df.to_csv(csv_path)

    # Data normalization
    # The values are divided by the median of the control in their group.
    caff_control_medians = {}
    ninej_control_medians = {}
    caff_control_df = pd.read_csv(os.path.join(
        target_path, 'caff_control_parameters.csv'), index_col=0)
    ninej_control_df = pd.read_csv(os.path.join(
        target_path, '9j_control_parameters.csv'), index_col=0)

    for param in params:
        caff_control_medians[param] = caff_control_df[param].median()
        ninej_control_medians[param] = ninej_control_df[param].median()

    group_list = glob.glob(os.path.join(target_path, '9j*.csv'))
    for file in tqdm(group_list):
        df_orig = pd.read_csv(file, index_col=0)
        for param in params:
            df_orig[param] = df_orig[param] / ninej_control_medians[param]
        df_orig.to_csv(os.path.join(target_path, os.path.basename(file)))

    group_list = glob.glob(os.path.join(target_path, 'caff*.csv'))
    for file in tqdm(group_list):
        df_orig = pd.read_csv(file, index_col=0)
        for param in params:
            df_orig[param] = df_orig[param] / caff_control_medians[param]
        df_orig.to_csv(os.path.join(target_path, os.path.basename(file)))

    csv_paths = list(set(glob.glob(os.path.join(target_path, '*_parameters.csv'))) - set(glob.glob(os.path.join(target_path, 'model*_parameters.csv'))))
    csv_paths.sort(reverse=True)
    csv_paths[0], csv_paths[1], csv_paths[2], csv_paths[3], csv_paths[4], csv_paths[5] = csv_paths[3], csv_paths[4], csv_paths[5], csv_paths[0], csv_paths[1], csv_paths[2]

    visual_target_path = os.path.join(os.getcwd(), 'data', 'original', 'visualization')
    os.makedirs(visual_target_path, exist_ok=True)

    columns = ['average_speed', 'total_distance', 'stop_duration', 'crosses_count', 'max_speed', 'sum_top_time', 'first_ascent_latency']
    columns_name = ['Average speed', 'Total distance', 'Stop duration', 'Crosses count', 'Max speed', 'Sum top time', 'First ascent latency']

    for ncol, col in enumerate(columns):
        df = pd.DataFrame(columns=['value', 'parameter', 'group', 'drug'])
        for csv_count, csv_path in enumerate(csv_paths):
            if csv_count == 0:
                csv_name = os.path.basename(csv_path)
                parametrs_df = pd.read_csv(csv_path, index_col=0).drop(['video_name'], axis=1)
                group = csv_name.replace('_parameters.csv', '').replace('9j_100mg', '++').replace('9j_1mg', '+').replace('9j_control', '0').replace('caff_control', '0').replace('caff_50mg', '+').replace('caff_100mg', '++')
                if csv_name.split('_')[0] == '9j':
                    drug = csv_name.split('_')[0]
                else:
                    drug = csv_name.split('_')[0]+'eine'
                for i in range(0, parametrs_df.shape[0]):
                        new_row = {'value': parametrs_df[col][i], 'parameter':col, 'group':group, 'drug':drug}
                        df = df.append(new_row, ignore_index=True)
            else:
                csv_name = os.path.basename(csv_path)
                new_parametrs_df = pd.read_csv(csv_path, index_col=0).drop(['video_name'], axis=1)
                new_parametrs_df['group']=csv_name.replace('_parameters.csv', '')
                group = csv_name.replace('_parameters.csv', '').replace('9j_100mg', '++').replace('9j_1mg', '+').replace('9j_control', '0').replace('caff_control', '0').replace('caff_50mg', '+').replace('caff_100mg', '++')
                if csv_name.split('_')[0] == '9j':
                    drug = csv_name.split('_')[0]
                else:
                    drug = csv_name.split('_')[0]+'eine'

                parametrs_df = pd.concat([new_parametrs_df, parametrs_df], ignore_index=True)
                for i in range(0, new_parametrs_df.shape[0]):
                        new_row = {'value':new_parametrs_df[col][i], 'parameter':col, 'group':group,  'drug':drug}
                        df = df.append(new_row, ignore_index=True)

        sns.set(font_scale=1.5, style='whitegrid')
        g = sns.catplot(x="group",
                        y="value",
                        hue="drug",
                        col = 'drug',
                        data=df,
                        kind="box",
                        width=0.7,
                        height=6,
                        showfliers=False,
                        aspect=.5,
                        palette=sns.color_palette('viridis', n_colors=2), dodge=False)

        g.set(xlabel=' ', ylabel='Fold of control')
        if ncol == 6:
            g.set(yscale="log")
        g.set_titles("{col_name}")
        title = columns_name[ncol]
        my_suptitle = g.figure.suptitle(title, fontsize=20, fontdict={"weight": "bold"}, y=1.025, x = 0.56)
        plt.figtext(0.35, 0.04, "Effector dose")
        plt.savefig(os.path.join(visual_target_path, col) + "_boxplot.svg", format = "svg", bbox_inches='tight', bbox_extra_artists=[my_suptitle])


def get_statistics_model(data_file: str, min_len: int, fps: float = 25, base: float = 1.05,
                               smin: int = 8, threshold_caff: float = 0.43301794443107317, threshold_9j: float = 0.8079598948655203) -> list:
    """
    Function for conventional analysis based on the estimation of multiple scalar metrics 

    Args:
        min_len(int): minimum trajectory length
        data_file(str): path to file for processing
        base(float):
        smin(int):

    Returns:
        data_list(list): list with parametrs in that order: [name, crossover position X, crossover position Y]
    """

    data = pd.read_csv(data_file, index_col=0)
    data = data.dropna(axis=0, how='all')
    data.index = range(0, len(data))

    data_size = data.shape[0]  # Clipping trajectories to the minimum length
    drop_size = data_size - min_len
    data = data.drop(list(range(data_size-drop_size, data_size)))

    data_list = []
    name = os.path.splitext(os.path.basename(data_file))[0]
    data_list.append(name)

    # DPCCA s_max for x и y
    L = min_len
    smax = L/4
    S = []
    for degree in range(int(math.log2(smin)/math.log2(base)), int(math.log2(smax)/math.log2(base))):
        new = int(base**degree)
        if new not in S:
            S.append(new)
    px, rx, fx, sx = dpcca(data['x'], pd=2, step=1, s=S, processes=2)
    new_f = fx*np.power(sx, -3/2)
    sx_max_index = np.argmax(new_f[int(len(new_f)/3):])
    sx_max = sx[sx_max_index]

    py, ry, fy, sy = dpcca(data['y'], pd=2, step=1, s=S, processes=2)
    new_f = fy*np.power(sy, -3/2)
    sy_max_index = np.argmax(new_f[int(len(new_f)/3):])
    sy_max = sy[sy_max_index]

    data_list.append(sx_max)
    data_list.append(sy_max)

    return data_list


def model_analysis(data_path: str, group_names: list, target_path: str):
    """
    Function for conventional analysis based on the estimation of multiple scalar metrics 

    Args:
        data_path(str): path to folder with trajectories
        group_names(list): list with experimental groups
        target_path(str): path for saving files

    Returns:
        saves csv file with patameters for groups to target_path
    """
    params = ['sx_max', 'sy_max']
    min_len = None
    for csv_path in glob.glob(os.path.join(data_path, '*.csv')):
        df = pd.read_csv(csv_path, index_col=0)
        df = df.dropna(axis=0, how='all')
        if min_len == None:
            min_len = df.shape[0]
        elif df.shape[0] < min_len:
            min_len = df.shape[0]

    for group_name in group_names:
        group_data_list = []
        for file_path in tqdm(glob.glob(os.path.join(data_path, group_name + '*.csv'))):
            group_data_list.append(
                get_statistics_model(file_path, min_len))

        group_data_df = pd.DataFrame(group_data_list, columns=['video_name', 'sx_max', 'sy_max'])

        # Outliers removing from DataFrame.
        # Outliers replaced to np.nan.
        for param in params:
            outliers_list = tukeys_method(group_data_df, param)
            for row_index in range(group_data_df.shape[0]):
                if group_data_df.loc[row_index, param] in outliers_list:
                    group_data_df.loc[row_index, param] = np.nan
        csv_path = os.path.join(target_path, 'model_' + group_name + '_parameters.csv')
        group_data_df.to_csv(csv_path)

    # Data normalization
    # The values are divided by the median of the control in their group.
    caff_control_medians = {}
    ninej_control_medians = {}
    caff_control_df = pd.read_csv(os.path.join(
        target_path, 'model_caff_control_parameters.csv'), index_col=0)
    ninej_control_df = pd.read_csv(os.path.join(
        target_path, 'model_9j_control_parameters.csv'), index_col=0)

    for param in params:
        caff_control_medians[param] = caff_control_df[param].median()
        ninej_control_medians[param] = ninej_control_df[param].median()

    group_list = glob.glob(os.path.join(target_path, 'model_9j*.csv'))
    for file in tqdm(group_list):
        df_orig = pd.read_csv(file, index_col=0)
        for param in params:
            df_orig[param + '_div_median'] = df_orig[param] / ninej_control_medians[param]
        df_orig.to_csv(os.path.join(target_path, os.path.basename(file)))

    group_list = glob.glob(os.path.join(target_path, 'model_caff*.csv'))
    for file in tqdm(group_list):
        df_orig = pd.read_csv(file, index_col=0)
        for param in params:
            df_orig[param + '_div_median'] = df_orig[param] / caff_control_medians[param]
        df_orig.to_csv(os.path.join(target_path, os.path.basename(file)))


    csv_paths = glob.glob(os.path.join(target_path, 'model_*_parameters.csv'))
    csv_paths.sort(reverse=True)
    csv_paths[0], csv_paths[1], csv_paths[2], csv_paths[3], csv_paths[4], csv_paths[5] = csv_paths[3], csv_paths[4], csv_paths[5], csv_paths[0], csv_paths[1], csv_paths[2]

    visual_target_path = os.path.join(os.getcwd(), 'data', 'original', 'visualization')
    os.makedirs(visual_target_path, exist_ok=True)

    columns = ['sx_max_div_median', 'sy_max_div_median']
    columns_name = ['Crossover position X', 'Crossover position Y']

    for ncol, col in enumerate(columns):
        df = pd.DataFrame(columns=['value', 'parameter', 'group', 'drug'])
        for csv_count, csv_path in enumerate(csv_paths):
            if csv_count == 0:
                csv_name = os.path.basename(csv_path)
                parametrs_df = pd.read_csv(csv_path, index_col=0).drop(['video_name'], axis=1)
                group = csv_name.replace('_parameters.csv', '').replace('model_9j_100mg', '++').replace('model_9j_1mg', '+').replace('model_9j_control', '0').replace('model_caff_control', '0').replace('model_caff_50mg', '+').replace('model_caff_100mg', '++')
                if csv_name.split('_')[1] == '9j':
                    drug = '9j'
                else:
                    drug = 'caffeine'
                for i in range(0,parametrs_df.shape[1]):
                        new_row = {'value':parametrs_df[col][i], 'parameter':col, 'group':group, 'drug':drug}
                        df = df.append(new_row, ignore_index=True)
            else:
                csv_name = os.path.basename(csv_path)
                new_parametrs_df = pd.read_csv(csv_path, index_col=0).drop(['video_name'], axis=1)
                new_parametrs_df['group']=csv_name.replace('_parameters.csv', '')
                group = csv_name.replace('_parameters.csv', '').replace('model_9j_100mg', '++').replace('model_9j_1mg', '+').replace('model_9j_control', '0').replace('model_caff_control', '0').replace('model_caff_50mg', '+').replace('model_caff_100mg', '++')
                if csv_name.split('_')[1] == '9j':
                    drug = csv_name.split('_')[1]
                else:
                    drug = csv_name.split('_')[1]+'eine'

                parametrs_df = pd.concat([new_parametrs_df, parametrs_df], ignore_index=True)
                for i in range(0, new_parametrs_df.shape[0]):
                        new_row = {'value':new_parametrs_df[col][i], 'parameter':col, 'group':group,  'drug':drug}
                        df = df.append(new_row, ignore_index=True)

        sns.set(font_scale =1.5, style='whitegrid')
        g = sns.catplot(x="group", y="value", hue="drug", col = 'drug', data=df, kind="box", width=0.7, height=6, showfliers = False, aspect=.5, palette=sns.color_palette('viridis', n_colors=2), dodge=False)
        g.set(xlabel = ' ', ylabel = 'Fold of control')
        if ncol == 6:
            g.set(yscale="log")
        g.set_titles("{col_name}")
        title = columns_name[ncol]
        my_suptitle = g.figure.suptitle(title, fontsize=20, fontdict={"weight": "bold"}, y=1.025, x = 0.56)
        plt.figtext(0.35, 0.04, "Effector dose")
        plt.savefig(os.path.join(visual_target_path, col) + "_boxplot.svg", format = "svg", bbox_inches='tight', bbox_extra_artists=[my_suptitle])


def model_trajectory_simulation(model_reg_file: str, traj_len: int, data_path: str, target_path: str, h: float = 1, length_degree: int = 15):
    """
        Function for trajectory simulation

        Args:
            model_reg_file(str): path file with regression
            traj_len(int): the length of the trajectory for modeling
            data_path(str): path to folder with parameters
            target_path(str): path to save model trajectories
            h(float): 
            length_degree(int): degree of 2 for trajectory length
    """

    csvs = glob.glob(os.path.join(data_path, 'model*.csv'))
    model_data = pd.read_csv(model_reg_file, index_col=0)
    model_data['crossover_position'] = pd.to_numeric(
        model_data['crossover_position'], downcast='integer')
    model_data['window_size'] = pd.to_numeric(model_data['window_size'])
    model_data['crossover_position'] = pd.to_numeric(
        model_data['crossover_position'])

    coef = np.polyfit(model_data['window_size'][0:20],
                      model_data['crossover_position'][0:20], 1)
    poly1d_fn = np.poly1d(1/coef)
    length = 2 ** length_degree

    for csv_path in csvs:
        data = pd.read_csv(csv_path, index_col=0)
        data = data.dropna(subset=['sx_max', 'sy_max'], axis=0)
        data.index = range(data.shape[0])
        group_name = os.path.basename(csv_path).replace('_parameters.csv', '')
        for row_index in tqdm(range(data.shape[0])):
            crossover_position = data['sx_max'][row_index]
            model_window_size = int(round(poly1d_fn(crossover_position), 0))
            if model_window_size == 1 or model_window_size == 0:
                continue
            noise = Filter(h, length).generate()
            x = movmean(noise, model_window_size)

            crossover_position = data['sy_max'][row_index]
            model_window_size = int(round(poly1d_fn(crossover_position), 0))
            if model_window_size == 1:
                continue
            noise = Filter(h, length).generate()
            y = movmean(noise, model_window_size)

            start_x = np.argmin(x)
            if start_x + traj_len > len(x):
                start_x = len(x) - traj_len
            end_x = start_x + traj_len

            start_y = np.argmax(y)
            if start_y + traj_len > len(y):
                start_y = len(y) - traj_len
            end_y = start_y + traj_len

            x = x[start_x: end_x]
            y = y[start_y: end_y]

            coords_df_x = pd.DataFrame(x, columns=['x'])
            coords_df_y = pd.DataFrame(y, columns=['y'])
            coords_df = pd.concat([coords_df_x, coords_df_y], axis=1)

            coords_df.to_csv(os.path.join(
                target_path, group_name.replace('model_', '') + '_' + str(row_index) + '.csv'))

