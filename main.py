import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from utils import generate_adj_list, generate_adj_matrix
from models import MyNetwork


def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parameters():
    parser = argparse.ArgumentParser(description='DConv-ST')
    parser.add_argument('--path_dir', type=str, default=r'E:\learn_pt2\graduation-program\project1\Data_PeMSD',
                        help='Data Root Directory')
    parser.add_argument('--dataset', type=str, default='PEMS08')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--num_of_recent', type=int, default=12)
    parser.add_argument('--num_of_days', type=int, default=7)
    parser.add_argument('--num_of_weeks', type=int, default=4)
    parser.add_argument('--num_for_predict', type=int, default=12)
    parser.add_argument('--drift_each_side', type=int, default=3)
    parser.add_argument('--index_of_feature', type=int, default=0,
                        help='{flow: 0, speed: 1, occupancy: 2}')
    parser.add_argument('--save_or_not', type=bool, default=False)
    parser.add_argument('--save_path', type=str,
                        default=r'Results_pems08/MyNetwork',
                        help='If save_or_not is True, save_path will be required.')
    parser.add_argument('--state', type=str, default='train', help='train or test')
    args = parser.parse_args()

    args.path_data = args.path_dir + '/' + args.dataset
    list_node_num = {'PEMS04': 307, 'PEMS08': 170}
    args.num_of_nodes = list_node_num[args.dataset]
    args.datafile_name = '{}_feat{}_{}-{}-{}in-{}out-{}drift.npz'.format(args.dataset, args.index_of_feature,
                                                                         args.num_of_recent, args.num_of_days,
                                                                         args.num_of_weeks, args.num_for_predict,
                                                                         args.drift_each_side)
    set_env(args.random_seed)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    blocks = {'mha_para': {'n_head': 4,
                           'd_k': 20,
                           'd_v': 16,
                           'd_x': 12,
                           'd_o': 12},
              'gat_para': {'in_c': 12,
                           'hid_c': 20,
                           'out_c': 12,
                           'n_heads': 2,
                           'alpha': 0.2},
              'temporal_para': {'in_c': 1,
                                'hid_c_list': [16, 4],
                                'out_c_list': [16, 12]}}

    blocks_fusion = {'spatio_para': {'n_nodes': args.num_of_nodes,
                                     'n_head_mha': 4,
                                     'd_k': 20,
                                     'd_v': 16,
                                     'd_x': 12,
                                     'n_heads_gat': 2,
                                     'alpha': 0.2,
                                     'in_c': 12,
                                     'hid_c': 20,
                                     'out_c': 12},
                     'temporal_para': {'in_c': 1,
                                       'hid_c_list': [16, 4],
                                       'out_c_list': [16, 12]},
                     'dconv_d_para': {'n_pred': args.num_for_predict,
                                      'n_drift': args.drift_each_side,
                                      'n_d': args.num_of_days,
                                      'n_nodes': args.num_of_nodes},
                     'dconv_w_para': {'n_pred': args.num_for_predict,
                                      'n_drift': args.drift_each_side,
                                      'n_w': args.num_of_weeks,
                                      'n_nodes': args.num_of_nodes},
                     'out_gating_para':{'in_c': args.num_of_recent,
                                        'out_c': args.num_for_predict,
                                        'n_nodes':args.num_of_nodes}}

    current_time = datetime.now().strftime("%D %H:%M")
    return args, blocks, blocks_fusion, current_time


def load_all_data(drop_last=False):
    adj_list = generate_adj_list(args.path_data + '/distance.csv')
    adj_matrix = generate_adj_matrix(adj_list, args.num_of_nodes)
    data = np.load(args.path_data + '/' + args.datafile_name)

    if args.state == 'train':
        train_dataset = TensorDataset(torch.tensor(data['xr_train'], dtype=torch.float32),
                                      torch.tensor(data['xd_train'], dtype=torch.float32),
                                      torch.tensor(data['xw_train'], dtype=torch.float32),
                                      torch.tensor(data['y_train'], dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(data['xr_eval'], dtype=torch.float32),
                                    torch.tensor(data['xd_eval'], dtype=torch.float32),
                                    torch.tensor(data['xw_eval'], dtype=torch.float32),
                                    torch.tensor(data['y_eval'], dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=drop_last, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=drop_last)
        return adj_list.to(args.device), adj_matrix.to(args.device), train_dataloader, val_dataloader

    elif args.state == 'test':
        test_dataset = TensorDataset(torch.tensor(data['xr_test'], dtype=torch.float32),
                                     torch.tensor(data['xd_test'], dtype=torch.float32),
                                     torch.tensor(data['xw_test'], dtype=torch.float32),
                                     torch.tensor(data['y_test'], dtype=torch.float32))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=drop_last)
        return adj_list.to(args.device), adj_matrix.to(args.device), test_dataloader


def model_initialize():
    model = MyNetwork(list_dconv_d=blocks_fusion['dconv_d_para'].values(),
                      list_dconv_w=blocks_fusion['dconv_w_para'].values(),
                      list_s=blocks_fusion['spatio_para'].values(),
                      list_t=blocks_fusion['temporal_para'].values(),
                      list_gating=blocks_fusion['out_gating_para'].values(),
                      resnet=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_criterion_mse = torch.nn.MSELoss().to(args.device)
    loss_criterion_mae = torch.nn.L1Loss().to(args.device)
    return model, optimizer, loss_criterion_mse, loss_criterion_mae


def train_epoch():
    epoch_training_mse = []
    epoch_training_mae = []
    num_data = 0  # Used to count how many pieces of data have been trained to calculate the total MSE for that round
    my_network.train()
    for batch_train_data in train_dataloader:
        optimizer.zero_grad()
        xr_batch, xd_batch, xw_batch, y_batch = batch_train_data
        xr_batch, xd_batch, xw_batch, y_batch = xr_batch.to(device=args.device), xd_batch.to(device=args.device), \
                                                xw_batch.to(device=args.device), y_batch.to(device=args.device)
        pred = my_network(xr_batch, xr_batch, xd_batch, xw_batch, adj_matrix)
        loss_mse = loss_criterion_mse(pred, y_batch)
        loss_mae = loss_criterion_mae(pred, y_batch)
        loss_mse.backward()
        optimizer.step()
        num_data += y_batch.shape[0]
        epoch_training_mse.append(loss_mse.detach().cpu().numpy() * y_batch.shape[0])
        epoch_training_mae.append(loss_mae.detach().cpu().numpy() * y_batch.shape[0])
    mse = sum(epoch_training_mse) / num_data
    mae = sum(epoch_training_mae) / num_data
    return mse, mae


def eval_test_epoch(dataloader):
    epoch_mse = []
    epoch_mae = []
    num_data = 0
    my_network.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            xr_batch, xd_batch, xw_batch, y_batch = batch_data
            xr_batch, xd_batch, xw_batch, y_batch = xr_batch.to(device=args.device), xd_batch.to(device=args.device), \
                                                    xw_batch.to(device=args.device), y_batch.to(device=args.device)
            pred = my_network(xr_batch, xr_batch, xd_batch, xw_batch, adj_matrix)
            loss_mse = loss_criterion_mse(pred, y_batch)
            loss_mae = loss_criterion_mae(pred, y_batch)
            epoch_mse.append(loss_mse.detach().cpu().numpy() * y_batch.shape[0])
            epoch_mae.append(loss_mae.detach().cpu().numpy() * y_batch.shape[0])
            num_data += y_batch.shape[0]
    mse = sum(epoch_mse) / num_data
    mae = sum(epoch_mae) / num_data
    return mse, mae


def save_model():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    pathname = os.path.join(args.save_path, 'model-epoch_{}.pth'.format(epoch + 1))
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': my_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, pathname)


if __name__ == '__main__':
    args, blocks, blocks_fusion, current_time = get_parameters()

    if args.state == 'train':
        print('Configs:***{}***{}--bs:{}--lr:{}--rs:{}--{}'.format
              (os.path.basename(__file__), args.dataset, args.batch_size, args.learning_rate, args.random_seed,
               current_time))
        adj_list, adj_matrix, train_dataloader, val_dataloader = load_all_data()

        my_network, optimizer, loss_criterion_mse, loss_criterion_mae = model_initialize()

        for epoch in range(args.num_epoch):
            train_mse, train_mae = train_epoch()
            val_mse, val_mae = eval_test_epoch(dataloader=val_dataloader)

            current_time = datetime.now().strftime("%D %H:%M")
            temp_print = "Epoch:{:2d}\t(Train)--RMSE:{:6.2f}\tMAE:{:6.2f}\t(Val)--RMSE:{:6.2f}\tMAE:{:6.2f}\tTime:{}" \
                .format(epoch + 1, np.sqrt(train_mse), train_mae, np.sqrt(val_mse), val_mae, current_time)
            print(temp_print)
            if args.save_or_not:
                save_model()
                with open(os.path.join(args.save_path, 'Train-Eval-Loss.txt'), 'a') as f:
                    f.write(temp_print + '\n')

    elif args.state == 'test':
        print('Configs:***{}***{}--bs:{}--lr:{}--rs:{}--{}'.format
              (os.path.basename(__file__), args.dataset, args.batch_size, args.learning_rate, args.random_seed,
               current_time))
        adj_list, adj_matrix, test_dataloader = load_all_data()
        my_network, optimizer, loss_criterion_mse, loss_criterion_mae = model_initialize()

        files = os.listdir(args.save_path)

        for i in range(args.num_epoch):
            file = 'model-epoch_{}.pth'.format(i + 1)
            file_path = os.path.join(args.save_path, file)

            checkpoint = torch.load(file_path)
            my_network.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            test_mse, test_mae = eval_test_epoch(dataloader=test_dataloader)

            print('Epoch:{:2d}\t(Test)--RMSE:{:6.2f}\tMAE:{:6.2f}'.format(i + 1, np.sqrt(test_mse), test_mae))
