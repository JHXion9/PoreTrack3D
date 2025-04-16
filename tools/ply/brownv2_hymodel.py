import torch
import torch.nn as nn
import torch.nn.functional as F
eps_l2_norm = 1e-10


class SDGMNet(nn.Module):
    """
    SDGMNet model adopted from Hynet 'HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss'(
    https://github.com/yuruntian/HyNet#hynet-learning-local-descriptor-with-hybrid-similarity-measure-and-triplet-loss).
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3):
        super(SDGMNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        
        self.features=nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7)
            
    def forward(self, x):
        feat = self.features(x)
        feat_t = feat.view(-1, self.dim_desc)
        feat_norm = F.normalize(feat_t, dim=1)
        return feat_norm


def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    # desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    if len(desc.shape) == 1:  # 如果只有一个维度
        desc = desc / desc.pow(2).sum().add(eps_l2_norm).pow(0.5).reshape(1,-1)  # 对所有元素求和
    else:
        desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    return desc

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        FRN layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """
        TLU layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class HyNet(nn.Module):
    """HyNet model definition
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc


class L2Net(nn.Module):
    """L2Net model definition
    """
    def __init__(self, is_bias=False, is_affine=False, dim_desc=128, drop_rate=0.3):
        super(L2Net, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1, affine=is_affine),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        return

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)

        return desc_l2norm(x.squeeze())


class HyNet_mixpre(nn.Module):
    """HyNet model definition
    """
    def __init__(self,concate_dim=4, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet_mixpre, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.concate_dim = concate_dim
        self.extra_dim  = int((32/concate_dim)**2)
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc-self.extra_dim, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc-self.extra_dim, affine=False)
        )
        
        self.premix = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.extra_dim, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.extra_dim, affine=False)
        )
        self.maxpoolpre = nn.MaxPool2d(kernel_size=self.concate_dim, stride=self.concate_dim)

    def forward(self, x, mode='eval'):
        xmaxpool = self.maxpoolpre(x)
        xmaxpool = xmaxpool.reshape(xmaxpool.shape[0],-1)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        xprimix  = self.premix(x).squeeze()
        extra_raw = xmaxpool*xprimix    
        extra = desc_l2norm(extra_raw)
        desc = desc_l2norm(desc_raw)
        desc_fina_raw = torch.cat([desc,extra],dim=1)
        desc_fina  = desc_l2norm(desc_fina_raw)
        if mode == 'train':
            return desc_fina, desc_fina_raw
        elif mode == 'eval':
            return desc_fina