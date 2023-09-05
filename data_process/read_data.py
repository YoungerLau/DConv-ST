import argparse
import configparser
import os.path

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


def process1_read_filename():
    """

    :return: len=7,  [Xr_input, Xr_timestamp, Xd_input, Xd_timestamp, Xw_input, Xw_timestamp, Xr_target]
    """
    filepath = os.path.join(out_path + dataset_name + '/process1/')
    file_list = []
    for type in ['input', 'timestamp']:
        filename = '{}_Xr_{}_feat{}_{}in-{}out.npz'.format(dataset_name, type, index_of_feature, num_of_recent,
                                                           num_for_predict)
        file_list.append(filepath + filename)
    for tag in ['Xd', 'Xw']:
        for type in ['input', 'timestamp']:
            filename = '{}_{}_{}_feat{}_{}in-{}out-{}drift.npz'.format(dataset_name, tag, type, index_of_feature,
                                                                       num_of_recent, num_for_predict, drift_each_side)
            file_list.append(filepath + filename)
    filename = '{}_Xr_target_feat{}_{}in-{}out.npz'.format(dataset_name, index_of_feature, num_of_recent, num_for_predict)
    file_list.append(filepath + filename)
    return file_list
