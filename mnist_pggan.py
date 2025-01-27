from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from progan_modules import EqualConvTranspose2d, PixelNorm, ConvBlock, EqualConv2d, upscale, EqualLinear, MnistConvBlock


class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=64, pixel_norm=True, tanh=True, use_mnist_conv_blocks=True):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.use_mnist_conv_blocks = use_mnist_conv_blocks
        self.pixel_norm = pixel_norm
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.1))

        conv_block = partial(MnistConvBlock) if use_mnist_conv_blocks else partial(ConvBlock)

        self.progression_4 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_8 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 1, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 1, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 1, 1)

        self.max_step = 3

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


class Discriminator(nn.Module):
    def __init__(self, feat_dim=64, use_mnist_conv_blocks=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_mnist_conv_blocks = use_mnist_conv_blocks
        conv_block = partial(MnistConvBlock) if use_mnist_conv_blocks else partial(ConvBlock)
        self.progression = nn.ModuleList([conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        # fixme: had to return to it because old models had it, so to successfully loading them those blocks
        # fixme: are unfortunately needed
        # fixme: REMOVE IT!
        self.mnist_progression_0 = MnistConvBlock(feat_dim + 1, feat_dim, 3, 1)
        self.mnist_progression_1 = MnistConvBlock(feat_dim + 1, feat_dim, 4, 0)

        self.from_rgb = nn.ModuleList([EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1)])

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
    def __init__(self, input_code_dim=128, num_of_classes=10, in_channel=64, pixel_norm=True, tanh=True,
                 use_mnist_conv_blocks=True
                 ):
        super().__init__()
        self.input_dim = input_code_dim
        self.in_channel = in_channel
        self.tanh = tanh
        self.use_mnist_conv_blocks = use_mnist_conv_blocks
        self.num_of_classes = num_of_classes
        self.embedding_dim = input_code_dim
        self.pixel_norm = pixel_norm
        self.embedding = nn.Embedding(num_of_classes, self.embedding_dim)
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim + self.embedding_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.1))

        conv_block = partial(MnistConvBlock) if use_mnist_conv_blocks else partial(ConvBlock)

        self.progression_4 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        # simple block with two convolutions, pixel norms and leaky relu
        self.progression_8 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = conv_block(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 1, 1)  # in_channel, out_channel, kernel_size
        self.to_rgb_16 = EqualConv2d(in_channel, 1, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 1, 1)

        self.max_step = 3

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

        out_4 = self.input_layer(data_in.view(-1, self.input_dim + self.embedding_dim, 1, 1))
        # here the latent vector is 'reshaped'
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


class ConditionalDiscriminatorWgangp(nn.Module):
    def __init__(self, feat_dim=64, num_of_classes=10, use_mnist_conv_blocks=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_of_classes = num_of_classes
        self.use_mnist_conv_blocks = use_mnist_conv_blocks
        conv_block = partial(MnistConvBlock) if use_mnist_conv_blocks else partial(ConvBlock)
        self.progression = nn.ModuleList([conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        # first attempt - prepare embeddings for every from_rgb to create just one more input dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_of_classes, 32**2),
            nn.Embedding(num_of_classes, 16**2),
            nn.Embedding(num_of_classes, 8**2),
            nn.Embedding(num_of_classes, 4**2),
        ])

        self.from_rgb = nn.ModuleList([EqualConv2d(1 + 1, feat_dim, 1),  # +1 because of the embedding
                                       EqualConv2d(1 + 1, feat_dim, 1),
                                       EqualConv2d(1 + 1, feat_dim, 1),
                                       EqualConv2d(1 + 1, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input_data, label, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                embedding = self.embeddings[index](label)
                embed_with_input = torch.cat([input_data, embedding.view(-1, 1, input_data.shape[-2], input_data.shape[-1])], 1)
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
                    skip_rgb = F.interpolate(input_data, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_embedding = self.embeddings[index + 1](label)
                    skip_rgb_with_embed = torch.cat([skip_rgb, skip_embedding.view(-1, 1, skip_rgb.shape[-2], skip_rgb.shape[-1])], 1)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb_with_embed)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)  # squeeze redundant dimensions
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out


class ConditionalDiscriminatorAda(nn.Module):
    def __init__(self, feat_dim=64, num_of_classes=10, use_mnist_conv_blocks=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_of_classes = num_of_classes
        self.embedding_dim = feat_dim
        self.use_mnist_conv_blocks = use_mnist_conv_blocks
        conv_block = partial(MnistConvBlock) if use_mnist_conv_blocks else partial(ConvBlock)
        self.progression = nn.ModuleList([conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          conv_block(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        # first attempt - prepare embeddings for every from_rgb to create just one more input dim
        self.embedding = nn.Embedding(num_of_classes, embedding_dim=self.embedding_dim)

        self.from_rgb = nn.ModuleList([EqualConv2d(1, feat_dim, 1),  # +1 because of the embedding
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1),
                                       EqualConv2d(1, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input_data, label, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input_data)

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
                    skip_rgb = F.interpolate(input_data, scale_factor=0.5, mode='bilinear', align_corners=False)
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
