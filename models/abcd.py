import functools
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary

###########################################################################################

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.ca = ChannelAttention(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.ca(x) * x
        out += residual
        out = self.relu(out)
        return out

############################################################################################

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):


        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.channel_attenton_block=BasicBlock(256)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]


        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out=self.conv_block(x)
        out=self.channel_attenton_block(out)
        out=x+out
        return out

# net=ResnetGenerator().to('cuda')
# summary(net,(3,256,256))


class G1(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d):
        super(G1, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class G2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):

        assert(n_blocks >= 0)
        super(G2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model=[]
        n_downsampling = 2

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)

class SD1(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(SD1, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        conv=spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        sequence = [conv, nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv1 = spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            sequence += [
                conv1,
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*sequence)


    def forward(self, input):
        return self.model(input)


class SD2(nn.Module):
    def __init__(self, input_nc=256, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(SD2, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1

        nf_mult = 4
        nf_mult_prev = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv2 = spectral_norm(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        sequence = [
            conv2,
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        conv3 = spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        sequence += [conv3]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)




# #加载预训练模型
# model=gen().to('cuda')
# model.load_state_dict(torch.load('../checkpoints/FFA_res_lp_lsal/latest_net_G.pth'))
# #自定义模型
# net=ge().to('cuda')
#
#
# dict={k: v for k, v in model.state_dict().items() if k in net.state_dict()}
# net.load_state_dict(dict)

