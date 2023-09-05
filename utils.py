import pandas as pd
import torch


def generate_adj_list(pathdir):
    """
    :param pathdir: Road network node relationship file (. csv), containing the first and last two nodes of all edges
    :return: (2, 2*num_edge), return results as an undirected graph
    """
    llist = [[], []]
    df = pd.read_csv(pathdir)
    for i in range(len(df)):
        llist[0].append(int(df.loc[i]['from']))
        llist[1].append(int(df.loc[i]['to']))
        llist[1].append(int(df.loc[i]['from']))
        llist[0].append(int(df.loc[i]['to']))
    llist = torch.tensor(llist, dtype=torch.long)
    return llist


def generate_adj_matrix(llist, num_of_nodes, loop=False):
    """
    :param llist: Result returned by function generate_adj_list(pathdir)
    :param num_of_nodes: Number of nodes
    :param loop: (Whether to consider the edges connected by nodes and themselves)
    :return: adj_matrix
    """
    adj = torch.zeros((num_of_nodes, num_of_nodes), dtype=torch.int32)
    for i in range(len(llist[0])):
        adj[int(llist[0][i])][int(llist[1][i])] = 1
    if loop:
        for i in range(num_of_nodes):
            adj[i][i] = 1
    return adj


