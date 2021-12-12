import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt


class EqualLR:  # Equalized learning rate
    """
    The idea behind equalized learning rate is to scale the weights at each layer with a constant such that the updated
    weight w’ is scaled to be w’ = w /c, where c is a constant at each layer. This is done during training to keep the
    weights in the network at a similar scale during training. This approach is unique because usually modern optimizers
    such as RMSProp and Adam use the standard deviation of the gradient to normalize it. This is problematic in the case
    where the weight is very large or small, in which case the standard deviation is an insufficient normalizer.

    Source: https://towardsdatascience.com/progressively-growing-gans-9cb795caebee
    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()  # todo: debug it to know what is happening here
        # numel func: Returns the total number of elements in the tensor.

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        # The hook will be called every time before forward() is invoked.
        # So before every forward() run this function will normalize weights of a passed module
        # with self.compute_weight() func.

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        # ConvTransposed2d args that goes in here (in_channels, out_channels, kernel_size, stride, padding)
        conv.weight.data.normal_()
        # tensor.normal_() will fill the tensor with values sampled from the normal distribution.
        # The old values will be overwritten.
        conv.bias.data.zero_()
        # same goes here I think
        self.conv = equal_lr(conv)
        # now add equalized learning rate

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class EqualEmbed(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EqualEmbed, self).__init__()
        embed = nn.Embedding(*args, **kwargs)
        embed.weight.data.normal_()
        self.embed = equal_lr(embed)

    def forward(self, input):
        return self.embed(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None,
                 pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.2))
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


class MnistConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, pixel_norm=True):
        super().__init__()

        convs = [EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)


class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=128, pixel_norm=True, tanh=True, max_step=6):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.pixel_norm = pixel_norm
        # todo: according to ProgressiveGAN the input layer which is represented here by input_layer
        # todo: and progression_4 should be like ConvBlock but the first
        # todo: conv block should be the one in below Sequential
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel // 2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel // 2, in_channel // 4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel // 4, in_channel // 4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel // 2, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel // 4, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel // 4, 3, 1)

        self.max_step = max_step

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # upscaling
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))  # here the latent vector is 'reshaped'
        # in some retarded fashion (maybe I'm retarded, that's possible) but it just adds dimensions and yet
        # the whole information is still describable by only one dimension
        out_4 = self.progression_4(out_4)
        # so the training basically looks that first we train the 4x4 and then automaticly 8x8
        # and after that we just start to do the progressive blending way of increasing the size
        # I wonder why they omitted this step between 4x4 and 8x8
        out_8 = self.progress(out_4, self.progression_8)
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.to_rgb_8(out_8)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 2:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 3:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)

        out_64 = self.progress(out_32, self.progression_64)
        if step == 4:
            return self.output(out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha)

        out_128 = self.progress(out_64, self.progression_128)
        if step == 5:
            return self.output(out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha)

        out_256 = self.progress(out_128, self.progression_256)
        if step == 6:
            return self.output(out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha)


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim

        self.progression = nn.ModuleList([ConvBlock(feat_dim // 4, feat_dim // 4, 3, 1),
                                          ConvBlock(feat_dim // 4, feat_dim // 2, 3, 1),
                                          ConvBlock(feat_dim // 2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, feat_dim // 4, 1),
                                       EqualConv2d(3, feat_dim // 4, 1),
                                       EqualConv2d(3, feat_dim // 2, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)  # not sure what tensor.var(0) means
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class ConditionalGenerator(nn.Module):
    def __init__(self, input_code_dim=128, num_of_classes=10, in_channel=128, pixel_norm=True, tanh=True, max_step=6):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.pixel_norm = pixel_norm
        self.num_of_classes = num_of_classes
        self.embedding_dim = num_of_classes
        self.embedding = nn.Embedding(num_of_classes, self.embedding_dim)
        # todo: according to ProgressiveGAN the input layer which is represented here by input_layer
        # todo: and progression_4 should be like ConvBlock but the first
        # todo: conv block should be the one in below Sequential
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim + self.embedding_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel // 2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel // 2, in_channel // 4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel // 4, in_channel // 4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel // 2, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel // 4, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel // 4, 3, 1)

        self.max_step = max_step

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # upscaling
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, label, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        # add embedding
        embed = self.embedding(label)
        data_in = torch.cat([input, embed], 1)

        out_4 = self.input_layer(data_in.view(-1, self.input_dim + self.embedding_dim, 1, 1))  # here the latent vector is 'reshaped'
        # in some retarded fashion (maybe I'm retarded, that's possible) but it just adds dimensions and yet
        # the whole information is still describable by only one dimension
        out_4 = self.progression_4(out_4)
        # so the training basically looks that first we train the 4x4 and then automaticly 8x8
        # and after that we just start to do the progressive blending way of increasing the size
        # I wonder why they omitted this step between 4x4 and 8x8
        out_8 = self.progress(out_4, self.progression_8)
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.to_rgb_8(out_8)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 2:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 3:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)

        out_64 = self.progress(out_32, self.progression_64)
        if step == 4:
            return self.output(out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha)

        out_128 = self.progress(out_64, self.progression_128)
        if step == 5:
            return self.output(out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha)

        out_256 = self.progress(out_128, self.progression_256)
        if step == 6:
            return self.output(out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha)


class ConditionalDiscriminatorWgangp(nn.Module):
    def __init__(self, feat_dim=128, num_of_classes=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_of_classes = num_of_classes

        self.progression = nn.ModuleList([ConvBlock(feat_dim // 4, feat_dim // 4, 3, 1),
                                          ConvBlock(feat_dim // 4, feat_dim // 2, 3, 1),
                                          ConvBlock(feat_dim // 2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_of_classes, 256**2),
            nn.Embedding(num_of_classes, 128**2),
            nn.Embedding(num_of_classes, 64**2),
            nn.Embedding(num_of_classes, 32**2),
            nn.Embedding(num_of_classes, 16**2),
            nn.Embedding(num_of_classes, 8**2),
            nn.Embedding(num_of_classes, 4**2),
        ])

        self.from_rgb = nn.ModuleList([EqualConv2d(3 + 1, feat_dim // 4, 1),
                                       EqualConv2d(3 + 1, feat_dim // 4, 1),
                                       EqualConv2d(3 + 1, feat_dim // 2, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, label, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                embedding = self.embeddings[index](label)
                embed_with_input = torch.cat([input, embedding.view(-1, 1, input.shape[-2], input.shape[-1])], 1)
                out = self.from_rgb[index](embed_with_input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)  # not sure what tensor.var(0) means
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_embedding = self.embeddings[index + 1](label)
                    skip_rgb_with_embed = torch.cat([skip_rgb, skip_embedding.view(-1, 1, skip_rgb.shape[-2], skip_rgb.shape[-1])], 1)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb_with_embed)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class CorrectGenerator(nn.Module):
    def __init__(self, input_code_dim=512, in_channel=512, pixel_norm=True, tanh=False, max_step=4):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.pixel_norm = pixel_norm

        self.progression_4 = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(in_channel, in_channel, 3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2)
        )

        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_4 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)

        self.max_step = max_step

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # upscaling
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.progression_4(input.view(-1, self.input_dim, 1, 1))
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_4(out_4))
            return self.to_rgb_4(out_4)

        out_8 = self.progress(out_4, self.progression_8)
        if step == 2:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.output(out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 3:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 4:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)


class CorrectDiscriminator(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.feat_dim = feat_dim

        self.progression = nn.ModuleList([
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, 0, -1):
            index = self.n_layer - i

            if i == step:
                out = self.from_rgb[index](input)

            if i == 1:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)  # not sure what tensor.var(0) means
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 1:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class ConditionalCorrectGenerator(nn.Module):
    def __init__(self, input_code_dim=512, num_of_classes=10, in_channel=512, pixel_norm=True, tanh=False, max_step=4,
                 do_equal_embed=False):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.pixel_norm = pixel_norm
        self.num_of_classes = num_of_classes
        self.embedding_dim = input_code_dim  # from ADA paper

        if do_equal_embed:
            self.embedding = EqualEmbed(num_embeddings=num_of_classes, embedding_dim=self.embedding_dim)
        else:
            self.embedding = nn.Embedding(num_of_classes, embedding_dim=self.embedding_dim)

        self.progression_4 = nn.Sequential(
            EqualConvTranspose2d(input_code_dim + self.embedding_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(in_channel, in_channel, 3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2)
        )

        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_4 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)

        self.max_step = max_step

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # upscaling
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, label, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        # add embedding
        embed = self.embedding(label)
        # concatenate normalized input and embedding
        # this idea was taken from ADA
        data_in = torch.cat([input, embed], 1)

        out_4 = self.progression_4(data_in.view(-1, self.input_dim + self.embedding_dim, 1, 1))
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_4(out_4))
            return self.to_rgb_4(out_4)

        out_8 = self.progress(out_4, self.progression_8)
        if step == 2:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.output(out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 3:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 4:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)


class ConditionalCorrectDiscriminatorWgangp(nn.Module):
    def __init__(self, feat_dim=128, num_of_classes=10, do_equal_embed=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_of_classes = num_of_classes

        self.progression = nn.ModuleList([
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])
        if do_equal_embed:
            self.embeddings = nn.ModuleList([
                EqualEmbed(num_embeddings=num_of_classes, embedding_dim=32**2),
                EqualEmbed(num_embeddings=num_of_classes, embedding_dim=16**2),
                EqualEmbed(num_embeddings=num_of_classes, embedding_dim=8**2),
                EqualEmbed(num_embeddings=num_of_classes, embedding_dim=4**2),
            ])
        else:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_of_classes, 32**2),
                nn.Embedding(num_of_classes, 16**2),
                nn.Embedding(num_of_classes, 8**2),
                nn.Embedding(num_of_classes, 4**2),
            ])

        self.from_rgb = nn.ModuleList([
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1),
                                       EqualConv2d(3 + 1, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, label, step=0, alpha=-1):
        for i in range(step, 0, -1):
            index = self.n_layer - i

            if i == step:
                embedding = self.embeddings[index](label)
                embed_with_input = torch.cat([input, embedding.view(-1, 1, input.shape[-2], input.shape[-1])], 1)
                out = self.from_rgb[index](embed_with_input)

            if i == 1:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)  # not sure what tensor.var(0) means
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 1:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_embedding = self.embeddings[index + 1](label)
                    skip_rgb_with_embed = torch.cat([skip_rgb, skip_embedding.view(-1, 1, skip_rgb.shape[-2], skip_rgb.shape[-1])], 1)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb_with_embed)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class ConditionalCorrectGeneratorAda(nn.Module):
    def __init__(self, input_code_dim=512, num_of_classes=10, in_channel=512, pixel_norm=True, tanh=False, max_step=4):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.pixel_norm = pixel_norm
        self.num_of_classes = num_of_classes
        self.embedding_dim = input_code_dim  # from ADA paper

        self.embedding = nn.Embedding(num_of_classes, embedding_dim=self.embedding_dim)

        self.progression_4 = nn.Sequential(
            EqualConvTranspose2d(input_code_dim + self.embedding_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(in_channel, in_channel, 3, padding=1),
            PixelNorm(),
            nn.LeakyReLU(0.2)
        )

        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_4 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)

        self.max_step = max_step

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # upscaling
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, label, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        # add embedding
        embed = self.embedding(label)
        # concatenate normalized input and embedding
        # this idea was taken from ADA
        data_in = torch.cat([torch.nn.functional.normalize(input), torch.nn.functional.normalize(embed)], 1)

        out_4 = self.progression_4(data_in.view(-1, self.input_dim + self.embedding_dim, 1, 1))
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_4(out_4))
            return self.to_rgb_4(out_4)

        out_8 = self.progress(out_4, self.progression_8)
        if step == 2:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.output(out_4, out_8, self.to_rgb_4, self.to_rgb_8, alpha)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 3:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 4:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)


class ConditionalCorrectDiscriminatorAda(nn.Module):
    def __init__(self, feat_dim=512, num_of_classes=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_of_classes = num_of_classes
        self.embedding_dim = feat_dim

        self.embedding = nn.Embedding(num_of_classes, embedding_dim=self.embedding_dim)

        self.progression = nn.ModuleList([
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)  # fix from pro_gan_pytorch repo

    def forward(self, input, label, step=0, alpha=-1):
        for i in range(step, 0, -1):
            index = self.n_layer - i

            if i == step:
                out = self.from_rgb[index](input)

            if i == 1:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)  # not sure what tensor.var(0) means
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 1:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        # add label information
        embed = torch.nn.functional.normalize(self.embedding(label))
        projection_scores = (out * embed).sum(dim=-1)
        out = self.linear(out)
        final_score = out.view(-1) + projection_scores

        return final_score
