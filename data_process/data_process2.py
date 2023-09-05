import argparse
import configparser
import os
import numpy as np
from read_data import process1_read_filename


def generate_final_data1(pro1_filename_list_, p_train, p_eval):
    xr_input = np.load(pro1_filename_list_[0])['input']
    xd_input = np.load(pro1_filename_list_[2])['input']
    xw_input = np.load(pro1_filename_list_[4])['input']
    xr_timestamp = np.load(pro1_filename_list_[1])['timestamp']
    xd_timestamp = np.load(pro1_filename_list_[3])['timestamp']
    xw_timestamp = np.load(pro1_filename_list_[5])['timestamp']
    xr_target = np.load(pro1_filename_list_[6])['target']
    target_points = len(xw_timestamp)
    xr_input_2 = xr_input[-target_points:]
    xd_input_2 = xd_input[-target_points:]
    xw_input_2 = xw_input[-target_points:]
    final_target = xr_target[-target_points:]

    print('Partitioning dataset...')
    split_line1 = int(target_points * p_train)
    split_line2 = int(target_points * (p_train + p_eval))
    xr_train, xd_train, xw_train = xr_input_2[0: split_line1], xd_input_2[0: split_line1], xw_input_2[
                                                                                           0: split_line1]
    y_train, y_eval, y_test = final_target[0: split_line1], final_target[split_line1: split_line2], final_target[
                                                                                                    split_line2:]

    print('Data standardization in progress1...')
    x_train_all_sample = np.concatenate((xr_train.flatten(), xd_train.flatten(), xw_train.flatten()))
    x_train_mean = x_train_all_sample.mean()
    x_train_std = x_train_all_sample.std()

    def normalize(x):
        return (x - x_train_mean) / x_train_std

    print('Data standardization in progress2...')
    xr_input_2_norm = normalize(xr_input_2)
    xd_input_2_norm = normalize(xd_input_2)
    xw_input_2_norm = normalize(xw_input_2)
    xr_train_norm, xd_train_norm, xw_train_norm = xr_input_2_norm[0: split_line1], xd_input_2_norm[
                                                                                   0: split_line1], xw_input_2_norm[
                                                                                                    0: split_line1]
    xr_eval_norm, xd_eval_norm, xw_eval_norm = xr_input_2_norm[split_line1: split_line2], xd_input_2_norm[
                                                                                          split_line1: split_line2], xw_input_2_norm[
                                                                                                                     split_line1: split_line2]
    xr_test_norm, xd_test_norm, xw_test_norm = xr_input_2_norm[split_line2:], xd_input_2_norm[
                                                                              split_line2:], xw_input_2_norm[
                                                                                             split_line2:]

    # all_data = {
    #     'xr': {
    #         'train': xr_train_norm,
    #         'eval': xr_eval_norm,
    #         'test': xr_test_norm
    #     },
    #     'xd': {
    #         'train': xd_train_norm,
    #         'eval': xd_eval_norm,
    #         'test': xd_test_norm
    #     },
    #     'xw': {
    #         'train': xw_train_norm,
    #         'eval': xw_eval_norm,
    #         'test': xw_test_norm
    #     },
    #     'y': {
    #         'train': y_train,
    #         'eval': y_eval,
    #         'test': y_test
    #     },
    #     'stats': {
    #         '_len': '{}-{}-{}, total:{}'.format(split_line1, split_line2 - split_line1, target_points - split_line2,
    #                                             target_points),
    #         '_mean': x_train_mean,
    #         '_std': x_train_std
    #     }
    # }
    #
    # print(all_data['stats']['_len'], all_data['stats']['_mean'], all_data['stats']['_std'])

    print('Saving data...')
    filename = '{}_feat{}_{}-{}-{}in-{}out-{}drift-process2.npz'.format(dataset_name, index_of_feature, num_of_recent,
                                                                        num_of_days, num_of_weeks,
                                                                        num_for_predict, drift_each_side)
    filepath = os.path.join(out_path + dataset_name + '/process2/' + filename)
    np.savez_compressed(filepath,
                        xr_train=xr_train_norm, xr_eval=xr_eval_norm, xr_test=xr_test_norm,
                        xd_train=xd_train_norm, xd_eval=xd_eval_norm, xd_test=xd_test_norm,
                        xw_train=xw_train_norm, xw_eval=xw_eval_norm, xw_test=xw_test_norm,
                        y_train=y_train, y_eval=y_eval, y_test=y_test,
                        _len='{}-{}-{}, total:{}'.format(split_line1, split_line2 - split_line1,
                                                         target_points - split_line2,
                                                         target_points),
                        _mean=x_train_mean,
                        _std=x_train_std)
    return None


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', type=str, default='./configuration/data.conf')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config_filename)
data_process1 = config['data_process1']
data_process2 = config['data_process2']
graph_signal_matrix_filename = data_process1['graph_signal_matrix_filename']
dataset_name = data_process1['dataset_name']
out_path = data_process1['out_path']
num_of_nodes = int(data_process1['num_of_nodes'])
num_of_recent = int(data_process1['num_of_recent'])
num_of_days = int(data_process1['num_of_days'])
num_of_weeks = int(data_process1['num_of_weeks'])
num_for_predict = int(data_process1['num_for_predict'])
drift_each_side = int(data_process1['drift_each_side'])
index_of_feature = int(data_process1['index_of_feature'])
points_per_day = int(data_process1['points_per_day'])
process1 = data_process1['process1']
process2 = data_process2['process2']
proportion_of_train = float(data_process2['proportion_of_train'])
proportion_of_eval = float(data_process2['proportion_of_eval'])

pro1_filename_list = process1_read_filename()
generate_final_data1(pro1_filename_list, proportion_of_train, proportion_of_eval)
