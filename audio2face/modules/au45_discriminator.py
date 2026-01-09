"""
AU45专用局部判别器
专门用于增强AU45（眨眼）的高频表现力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class AU45LocalDiscriminator(nn.Module):
    """
    AU45专用局部判别器
    设计用于捕捉AU45的高频细节和时序特性
    """
    def __init__(self, input_nc=1, ndf=32, n_layers=2, norm_layer=nn.BatchNorm1d):
        super(AU45LocalDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        
        # 1D卷积用于时序建模
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=5, stride=2, padding=2), 
                   nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=5, stride=2, 
                         padding=2, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, 
                     padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层：每个时间步一个预测
        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input, tmask=None):
        """
        Args:
            input: [B, T] AU45序列
            tmask: [B, T] 可选的时间掩码（会在输出长度上匹配）
        """
        # 添加通道维度: [B, T] -> [B, 1, T]
        if input.dim() == 2:
            input = input.unsqueeze(1)
        
        output = self.model(input)  # [B, 1, T_out]，T_out可能小于T_in（由于stride）
        
        # 如果有时间掩码，应用掩码（需要匹配输出长度）
        if tmask is not None:
            # 获取输出长度
            output_len = output.shape[2]
            tmask_len = tmask.shape[1] if tmask.dim() == 2 else tmask.shape[2]
            
            # 如果mask长度与输出长度不匹配，进行插值或裁剪
            if tmask.dim() == 2:
                tmask = tmask.unsqueeze(1)  # [B, 1, T]
            
            if tmask.shape[2] != output_len:
                # 使用插值将mask调整到输出长度
                tmask = F.interpolate(tmask, size=output_len, mode='nearest')
            
            output = output * tmask
        
        return output.squeeze(1)  # [B, T_out]
    
    def extract_features(self, input):
        """
        提取中间层特征用于特征匹配
        """
        if input.dim() == 2:
            input = input.unsqueeze(1)
        
        features = []
        x = input
        for i, layer in enumerate(self.model):
            x = layer(x)
            # 提取中间层特征（在LeakyReLU之后）
            if isinstance(layer, nn.LeakyReLU) and i < len(self.model) - 2:
                features.append(x)
        
        return features

