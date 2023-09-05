import configparser
import argparse
import os
import numpy as np


def generate_xr_input_or_target(graph_signal_matrix_filename, timestamp_list,
                                num_of_nodes, num_of_recent, num_for_predict, index_of_feature):
    assert (num_of_recent * num_for_predict == 0) and (num_of_recent + num_for_predict > 0)
    rawdata = np.load(graph_signal_matrix_filename)['data']
    llist = [[] for i in range(len(timestamp_list))]
    if num_for_predict == 0:
        for timestamp_old in timestamp_list:
            for node in range(num_of_nodes):
                timestamp_new = timestamp_old - num_of_recent
                llist[timestamp_new].append(rawdata[timestamp_new: timestamp_old, node, index_of_feature])
        return np.array(llist)
    elif num_of_recent == 0:
        timestamp_new = 0
        for timestamp_old in timestamp_list:
            for node in range(num_of_nodes):
                llist[timestamp_new].append(
                    rawdata[timestamp_old: (timestamp_old + num_for_predict), node, index_of_feature])
            timestamp_new += 1
        return np.array(llist)


def generate_xd_or_xw_input(graph_signal_matrix_filename, timestamp_list, num_of_nodes, points_per_day,
                            tag, num_of_dimension, drift_each_side, num_for_predict, index_of_feature):
    rawdata = np.load(graph_signal_matrix_filename)['data']
    llist = [[[] for j in range(num_of_nodes)] for i in range(len(timestamp_list))]
    interval_dict = {'Xd': 1, 'Xw': 7}
    interval_days = interval_dict[tag]
    timestamp_new = 0
    for timestamp_old in timestamp_list:
        for node in range(num_of_nodes):
            for day in range(num_of_dimension):
                key_stamp = timestamp_old - points_per_day * (num_of_dimension - day) * interval_days
                llist[timestamp_new][node].append(
                    rawdata[(key_stamp - drift_each_side):(key_stamp + num_for_predict + drift_each_side),
                    node, index_of_feature])
        timestamp_new += 1
    return np.array(llist)


def generate_xr_data(graph_signal_matrix_filename, out_path, dataset_name,
                     num_of_nodes, num_of_recent, num_for_predict, index_of_feature):
    filename1 = '{}_Xr_input_feat{}_{}in-{}out.npz'.format(dataset_name, index_of_feature, num_of_recent,
                                                           num_for_predict)
    filename2 = '{}_Xr_timestamp_feat{}_{}in-{}out.npz'.format(dataset_name, index_of_feature, num_of_recent,
                                                               num_for_predict)
    filename3 = '{}_Xr_target_feat{}_{}in-{}out.npz'.format(dataset_name, index_of_feature, num_of_recent,
                                                            num_for_predict)
    filepath1 = os.path.join(out_path + dataset_name + '/' + filename1)
    filepath2 = os.path.join(out_path + dataset_name + '/' + filename2)
    filepath3 = os.path.join(out_path + dataset_name + '/' + filename3)
    if os.path.exists(filepath1):
        temp = input("The file already exists. Generate it again？(y/n)")
        if temp == 'n':
            exit()
    len_of_rawdata = np.load(graph_signal_matrix_filename)['data'].shape[0]
    timestamp_list = np.arange(num_of_recent, len_of_rawdata - num_for_predict + 1)
    len_of_time = len(timestamp_list)
    np.savez_compressed(filepath2, len=len_of_time, timestamp=timestamp_list)
    xr_input = generate_xr_input_or_target(graph_signal_matrix_filename, timestamp_list, num_of_nodes, num_of_recent, 0,
                                           index_of_feature)
    np.savez_compressed(filepath1, input=xr_input)  # (16969, 307, 12)
    xr_target = generate_xr_input_or_target(graph_signal_matrix_filename, timestamp_list, num_of_nodes, 0,
                                            num_for_predict, index_of_feature)
    np.savez_compressed(filepath3, target=xr_target)
    return None


def generate_xd_or_xw_data(graph_signal_matrix_filename, out_path, dataset_name,
                           num_of_nodes, num_of_days, num_of_weeks, drift_each_side, num_for_predict, index_of_feature):
    assert (num_of_days * num_of_weeks == 0) and (num_of_days + num_of_weeks > 0)
    if num_of_weeks == 0:
        num_of_which = num_of_days
        tag = 'Xd'
    elif num_of_days == 0:
        num_of_which = num_of_weeks * 7
        tag = 'Xw'
    filename_input = '{}_{}_input_feat{}_{}in-{}out-{}drift.npz'.format(dataset_name, tag, index_of_feature,
                                                                        num_of_recent, num_for_predict, drift_each_side)
    filename_timestamp = '{}_{}_timestamp_feat{}_{}in-{}out-{}drift.npz'.format(dataset_name, tag, index_of_feature,
                                                                                num_of_recent, num_for_predict,
                                                                                drift_each_side)
    filepath_input = os.path.join(out_path + dataset_name + '/' + filename_input)
    filepath_timestamp = os.path.join(out_path + dataset_name + '/' + filename_timestamp)
    len_of_rawdata = np.load(graph_signal_matrix_filename)['data'].shape[0]
    timestamp_list = np.arange(drift_each_side + num_of_which * points_per_day, len_of_rawdata - num_for_predict + 1)
    len_of_time = len(timestamp_list)
    np.savez_compressed(filepath_timestamp, len=len_of_time, timestamp=timestamp_list)
    x_input = generate_xd_or_xw_input(graph_signal_matrix_filename, timestamp_list, num_of_nodes, points_per_day,
                                      tag, num_of_days + num_of_weeks, drift_each_side, num_for_predict,
                                      index_of_feature)
    np.savez_compressed(filepath_input, input=x_input)
    return None


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', type=str, default='./configuration/data.conf')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config_filename)
data_process1 = config['data_process1']
graph_signal_matrix_filename = data_process1['graph_signal_matrix_filename']
dataset_name = data_process1['dataset_name'] + data_process1['process1']  # notice
out_path = data_process1['out_path']
num_of_nodes = int(data_process1['num_of_nodes'])
num_of_recent = int(data_process1['num_of_recent'])
num_of_days = int(data_process1['num_of_days'])
num_of_weeks = int(data_process1['num_of_weeks'])
num_for_predict = int(data_process1['num_for_predict'])
drift_each_side = int(data_process1['drift_each_side'])
index_of_feature = int(data_process1['index_of_feature'])
points_per_day = int(data_process1['points_per_day'])

generate_xr_data(graph_signal_matrix_filename, out_path, dataset_name, num_of_nodes, num_of_recent, num_for_predict, index_of_feature)
generate_xd_or_xw_data(graph_signal_matrix_filename, out_path, dataset_name,
                       num_of_nodes, num_of_days, 0, drift_each_side, num_for_predict, index_of_feature)
generate_xd_or_xw_data(graph_signal_matrix_filename, out_path, dataset_name,
                       num_of_nodes, 0, num_of_weeks, drift_each_side, num_for_predict, index_of_feature)
