from layers import *


class Gating(nn.Module):
    def __init__(self, in_c, out_c, n_nodes):
        """
        :param in_c: n_recent
        :param out_c: n_pred
        :param n_nodes:
        """
        super(Gating, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_c, out_c))
        self.b = nn.Parameter(torch.Tensor(n_nodes, out_c))
        nn.init.normal_(self.W)
        nn.init.normal_(self.b)
        self.F = torch.sigmoid

    def forward(self, ctrl, inp1, inp2):
        activate = self.F(torch.matmul(ctrl, self.W) + self.b)
        out = torch.mul(activate, inp1) + torch.mul(1 - activate, inp2)
        return out


class SpaceFusionGating(nn.Module):
    def __init__(self, n_nodes, n_head_mha, d_k, d_v, d_x, n_heads_gat, alpha, in_c, hid_c, out_c):
        super(SpaceFusionGating, self).__init__()
        self.mha = SelfAttention(n_head_mha, d_k, d_v, d_x, out_c)
        self.gat = GATNet(in_c, hid_c, out_c, n_heads_gat, alpha)
        self.G = Gating(in_c, out_c, n_nodes)

    def forward(self, ctrl, inp, adj):  # Using xr as gated control
        out_mha = self.mha(inp)
        out_gat = self.gat(inp, adj)
        output = self.G(ctrl, out_mha, out_gat)
        return output


class STFusionResEach(nn.Module):
    """
    The time and space layers in each spatiotemporal fusion module adopt residual networks
    """
    def __init__(self, list_s, list_t, resnet):
        super(STFusionResEach, self).__init__()
        self.spatio_fusion = SpaceFusionGating(*list_s)
        self.temporal_fusion = TemporalConvNet(*list_t)
        self.resnet = resnet

    def forward(self, ctrl, inp, adj):
        out_s = self.spatio_fusion(ctrl, inp, adj)
        if self.resnet:
            out_s = out_s + inp
        b, n, t = out_s.shape[0], out_s.shape[1], out_s.shape[2]
        inp_t = out_s.reshape(b * n, 1, t)  # -> (B*N, 1, T)
        out_t = self.temporal_fusion(inp_t)  # -> (B*N, out_c)
        bn = out_t.shape[0]
        out_t = out_t.reshape(int(bn / n), n, -1)  # -> (B, N, out_c)
        if self.resnet:
            out_t = out_t + out_s
        return out_t


class TwoSTLayersWithResEach(nn.Module):
    """
    Two layer STFusionResEach, used for the model of xr
    """
    def __init__(self, list_s, list_t, resnet):
        super(TwoSTLayersWithResEach, self).__init__()
        self.st1 = STFusionResEach(list_s, list_t, resnet)
        self.st2 = STFusionResEach(list_s, list_t, resnet)

    def forward(self, ctrl, inp, adj):
        out1 = self.st1(ctrl, inp, adj)
        out2 = self.st2(ctrl, out1, adj)
        return out2


class DConvST(nn.Module):
    def __init__(self, list_dconv, list_s, list_t, resnet):
        """
        :param list_dconv: list_dconv_d = [n_pred, n_drift, n_d, n_nodes], list_dconv_w is similar to list_dconv_d
        :param list_s:
        :param list_t:
        :param resnet: True, set whether to add residuals through this parameter
        """
        super(DConvST, self).__init__()
        self.dconv = DeformableConvStandard(*list_dconv)
        self.st = STFusionResEach(list_s, list_t, resnet)

    def forward(self, ctrl, x_dw, adj):  # ctrl param is xr
        out = self.dconv(x_dw, ctrl)
        out = self.st(ctrl, out, adj)
        return out


class MyNetwork(nn.Module):
    def __init__(self, list_dconv_d, list_dconv_w, list_s, list_t, list_gating, resnet):
        super(MyNetwork, self).__init__()
        self.xr_model = TwoSTLayersWithResEach(list_s, list_t, resnet)
        self.xd_model = DConvST(list_dconv_d, list_s, list_t, resnet)
        self.xw_model = DConvST(list_dconv_w, list_s, list_t, resnet)
        self.G1 = Gating(*list_gating)
        self.G2 = Gating(*list_gating)

    def forward(self, ctrl, inp_r, inp_d, inp_w, adj):
        out_r = self.xr_model(ctrl, inp_r, adj)
        out_d = self.xd_model(ctrl, inp_d, adj)
        out_w = self.xw_model(ctrl, inp_w, adj)

        out_dw = self.G1(ctrl, out_d, out_w)
        out = self.G2(ctrl, out_dw, out_r)

        return out
