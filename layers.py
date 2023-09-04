import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask==0, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head * batch, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super(SelfAttention, self).__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):  # mask.shape = (d_x, d_x)
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.alpha = alpha

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))
        self.a = nn.Parameter(torch.Tensor(2 * out_c, 1))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)
        nn.init.normal_(self.a)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, T].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]
        B = h.size()[0]
        N = h.size()[1]
        outputs = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_c)
        outputs = self.leakyrelu(torch.matmul(outputs, self.a).squeeze(3)) * graph
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))  # Make the places with 0 in outputs very small negative values
        attention = self.F(outputs, dim=2)  # [B, N, N]
        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D]


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads, alpha):
        super(GATSubNet, self).__init__()
        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c, alpha) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c, alpha)
        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, T]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)
        outputs = self.out_att(outputs, graph)
        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads, alpha):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads, alpha)
        # self.subnet = [GATSubNet(...) for _ in range(T)]

    def forward(self, flow, graph):
        """
        :param flow: (B, N, T)
        :param graph: (N, N),(B, N, N)
        :return:
        """
        b, n = flow.size(0), flow.size(1)
        assert graph.size(0) == n, "Dimension mismatch!"
        prediction = self.subnet(flow, graph.repeat(b, 1, 1)).squeeze(2)  # [B, N, 1, T] -> [B, N, T]
        return prediction


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 方便ResidualNET
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, kernel_size=3, dropout=0.2):
        """
        :param num_inputs:
        :param num_channels:
        :param num_outputs: out_c_list, [ , 12], Linear1-ReLU-Linear2
        :param kernel_size:
        :param dropout:
        """
        super(TemporalConvNet, self).__init__()
        self.linear1 = nn.Linear(num_channels[-1], num_outputs[0])
        self.linear2 = nn.Linear(num_outputs[0], num_outputs[1])
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (B*N, 1, T)
        :return: (B*N, out_c)
        """
        pred = self.network(x)[:, :, -1]
        pred = self.linear1(pred)
        pred = F.relu(pred)
        pred = self.linear2(pred)
        return pred


class STFusion(nn.Module):
    def __init__(self, list_s, list_t):
        """
        spatio->temporal
        :param list_s: [n_head, d_k, d_v, d_x, d_o]
        :param list_t: [in_c, hid_c:list, out_c:list[ , ]]
        """
        super(STFusion, self).__init__()
        self.spatio_fusion = SelfAttention(*list_s)  # 12, 20, 12, 307, 4, 0.2
        self.temporal_fusion = TemporalConvNet(*list_t)  # 1, [16, 4], [16, 12]

    def forward(self, inp):
        out = self.spatio_fusion(inp, mask=None)  # -> (B, N, T)
        b, n, t = out.shape[0], out.shape[1], out.shape[2]
        out = out.reshape(b * n, 1, t)  # -> (B*N, 1, T)
        out = self.temporal_fusion(out)  # -> (B*N, out_c)
        bn = out.shape[0]
        out = out.reshape(int(bn/n), n, -1)  # -> (B, N, out_c)
        return out


class DeformableConvStandard(nn.Module):    # Deformable Convolution Module
    def __init__(self, n_pred, n_drift, n_d_w, n_nodes):
        super(DeformableConvStandard, self).__init__()
        self.n_pred = n_pred
        self.n_drift = n_drift
        self.n_d_w = n_d_w

        self.conv1d_t = nn.Conv1d(n_d_w, 1, kernel_size=3, stride=1, padding=1)
        self.offset_t = nn.Parameter(torch.zeros((n_d_w, n_pred)), requires_grad=True)
        self.pos_add_t = nn.Parameter(torch.tensor([[i + n_drift for i in range(n_pred)] for _ in range(n_d_w)],
                                                 dtype=torch.int32, requires_grad=False), requires_grad=False)

        self.conv1d_n = nn.Conv1d(n_pred, n_pred, kernel_size=n_d_w)
        self.offset_n = nn.Parameter(torch.zeros((n_d_w, n_pred)), requires_grad=True)
        self.pos_add_n = nn.Parameter(torch.tensor([[i + n_drift for i in range(n_pred)] for _ in range(n_d_w)],
                                                   dtype=torch.int32, requires_grad=False), requires_grad=False)

        self.W = nn.Parameter(torch.Tensor(n_pred, n_pred))
        self.b = nn.Parameter(torch.Tensor(n_nodes, n_pred))
        nn.init.normal_(self.W)
        nn.init.normal_(self.b)
        self.F = torch.sigmoid

    def one_d_interpolation_3shape_t(self, i, j, input_data):  # Interpolation function for t
        offset_activate = torch.tanh(self.offset_t) * self.n_drift + self.pos_add_t
        pos = offset_activate[i][j]
        key = int(pos)
        return input_data[:, i, key] * (1 - pos + key) + input_data[:, i, key + 1] * (pos - key)

    def one_d_interpolation_3shape_n(self, i, j, input_data):  # Interpolation function for n
        offset_activate = torch.tanh(self.offset_n) * self.n_drift + self.pos_add_n
        pos = offset_activate[i][j]
        key = int(pos)
        return input_data[:, i, key] * (1 - pos + key) + input_data[:, i, key + 1] * (pos - key)

    def forward(self, inp, ctrl):
        b, n = inp.shape[0], inp.shape[1]
        inp = inp.view(b * n, self.n_d_w, -1)  # (B, N, D, Tp+2*drift)->(B*N, D, Tp+2*drift)
        inp_deform_t = inp.clone()
        inp_deform_n = inp.clone()
        for i in range(self.n_d_w):
            for j in range(self.n_pred):
                inp_deform_t[:, i, j + self.n_drift] = self.one_d_interpolation_3shape_t(i, j, inp)
                inp_deform_n[:, i, j + self.n_drift] = self.one_d_interpolation_3shape_n(i, j, inp)

        inp_deform_t = inp_deform_t[:, :, self.n_drift:(-self.n_drift)].contiguous()
        # (B*N, D, Tp+2*drift) -> (B*N, D, Tp)
        pred_t = self.conv1d_t(inp_deform_t).view(b, n, -1)
        # -> (B*N, 1, Tp) -> (B, N, Tp)

        inp_deform_n = inp_deform_n[:, :, self.n_drift:(-self.n_drift)].permute(0, 2, 1).contiguous()
        # (B*N, D, Tp+2*drift) -> (B*N, D, Tp) -> (B*N, Tp, D)
        pred_n = self.conv1d_n(inp_deform_n).view(b, n, -1)
        # -> (B*N, Tp, 1) -> (B, N, Tp)

        activate = torch.matmul(ctrl, self.W) + self.b
        out = torch.mul(pred_n, self.F(activate)) + torch.mul(pred_t, (1 - self.F(activate)))
        return out
