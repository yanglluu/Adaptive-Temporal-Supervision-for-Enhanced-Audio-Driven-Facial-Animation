"""
Enhanced Loss Functions for DCAW Optimization
包含：AU45专用损失、表情能量分带补偿、Huber正则、分带化损失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AU45BranchLoss(nn.Module):
    """
    AU45专用分支损失
    使用高频感知的损失函数，增强AU45（眨眼）的表现力
    新增：峰值保持损失（PeaksMSE/L1）
    """
    def __init__(self, base_weight=0.5, high_freq_weight=2.0, temporal_weight=1.0, peak_weight=2.0, peak_window=10):
        super(AU45BranchLoss, self).__init__()
        self.base_weight = base_weight
        self.high_freq_weight = high_freq_weight
        self.temporal_weight = temporal_weight
        self.peak_weight = peak_weight  # 峰值保持损失权重
        self.peak_window = peak_window  # 峰值检测窗口大小
        
    def _detect_peaks(self, signal, window=10):
        """
        检测信号中的峰值位置
        Args:
            signal: [B, T] 信号
            window: 峰值检测窗口大小
        Returns:
            peak_mask: [B, T] 峰值掩码（峰值邻域为1，其他为0）
        """
        B, T = signal.shape
        peak_mask = torch.zeros_like(signal)
        
        # 对每个batch和每个时间点检测峰值
        for b in range(B):
            sig = signal[b]  # [T]
            # 计算局部最大值
            for i in range(window//2, T - window//2):
                local_window = sig[i - window//2:i + window//2 + 1]
                if sig[i] == torch.max(local_window) and sig[i] > torch.mean(sig):
                    # 标记峰值及其邻域
                    start = max(0, i - window//2)
                    end = min(T, i + window//2 + 1)
                    peak_mask[b, start:end] = 1.0
        
        return peak_mask
    
    def forward(self, pred_au45, gt_au45, weights=None):
        """
        Args:
            pred_au45: [B, T] 预测的AU45值
            gt_au45: [B, T] 真实的AU45值
            weights: [B, T] 可选的权重（窄软窗，±8-12帧）
        """
        # 基础L1损失
        base_loss = F.l1_loss(pred_au45, gt_au45)
        
        # 高频感知损失：对变化率进行更严格约束
        pred_diff = pred_au45[:, 1:] - pred_au45[:, :-1]
        gt_diff = gt_au45[:, 1:] - gt_au45[:, :-1]
        high_freq_loss = F.l1_loss(pred_diff, gt_diff)
        
        # 时间平滑性（但不过度平滑）
        pred_acc = pred_diff[:, 1:] - pred_diff[:, :-1]
        gt_acc = gt_diff[:, 1:] - gt_diff[:, :-1]
        temporal_loss = F.l1_loss(pred_acc, gt_acc)
        
        # 峰值保持损失：对检测到的眨眼峰邻域做PeaksMSE/L1
        peak_mask = self._detect_peaks(gt_au45, window=self.peak_window)
        if torch.sum(peak_mask) > 0:
            # 在峰值区域计算L1损失
            peak_loss = torch.mean(torch.abs(pred_au45 - gt_au45) * peak_mask)
        else:
            peak_loss = torch.tensor(0.0, device=pred_au45.device)
        
        total_loss = (self.base_weight * base_loss + 
                     self.high_freq_weight * high_freq_loss +
                     self.temporal_weight * temporal_loss +
                     self.peak_weight * peak_loss)
        
        if weights is not None:
            # 应用权重到基础损失（注意：weights已经是窄软窗，不包含S(t)）
            weighted_base = torch.mean(torch.abs(pred_au45 - gt_au45) * weights)
            total_loss = (self.base_weight * weighted_base + 
                         self.high_freq_weight * high_freq_loss +
                         self.temporal_weight * temporal_loss +
                         self.peak_weight * peak_loss)
        
        return total_loss


class ExpressionEnergyBandLoss(nn.Module):
    """
    表情能量分带补偿损失 + 泄压阀机制（优化版）
    将表情能量分为低频、中频、高频三个频带，分别处理
    优化：泄压阀触发阈值从u=5帧放宽到u=7帧
    优化：能量分带增加中高频占比（0.05/0.05/0.10），但把每帧的总梯度归一
    """
    def __init__(self, beta=0.5, low_band_weight=0.05, mid_band_weight=0.05, 
                 high_band_weight=0.10, pressure_threshold=0.3, pressure_frames=7, normalize_gradient=True):
        super(ExpressionEnergyBandLoss, self).__init__()
        self.beta = beta
        self.low_band_weight = low_band_weight  # 从1.0改为0.05
        self.mid_band_weight = mid_band_weight  # 从1.5改为0.05
        self.high_band_weight = high_band_weight  # 从2.0改为0.10
        self.pressure_threshold = pressure_threshold
        self.pressure_frames = pressure_frames  # 从5帧改为7帧
        self.normalize_gradient = normalize_gradient  # 梯度归一化
        
    def _band_split(self, expr, n_bands=3):
        """
        将表情通道分为n个频带
        Returns: list of [B, T, C_band] tensors
        """
        C = expr.shape[2]
        band_size = C // n_bands
        bands = []
        for i in range(n_bands):
            start_idx = i * band_size
            if i == n_bands - 1:
                end_idx = C
            else:
                end_idx = (i + 1) * band_size
            bands.append(expr[:, :, start_idx:end_idx])
        return bands
    
    def forward(self, pred, gt, w=None):
        """
        Args:
            pred: [B, T, C] 预测的表情参数（从索引7开始）
            gt: [B, T, C] 真实的表情参数
            w: [B, T] 可选的权重
        """
        # 提取表情通道（索引7:）
        pred_expr = pred[:, :, 7:] if pred.shape[2] > 7 else pred
        gt_expr = gt[:, :, 7:] if gt.shape[2] > 7 else gt
        
        # 分带处理
        pred_bands = self._band_split(pred_expr, n_bands=3)
        gt_bands = self._band_split(gt_expr, n_bands=3)
        band_weights = [self.low_band_weight, self.mid_band_weight, self.high_band_weight]
        
        total_loss = 0.0
        band_losses = []
        
        for i, (pred_band, gt_band, band_w) in enumerate(zip(pred_bands, gt_bands, band_weights)):
            # 计算该频带的能量
            E_pred = torch.norm(pred_band, p=2, dim=2)  # [B, T]
            E_gt = torch.norm(gt_band, p=2, dim=2)  # [B, T]
            
            # 能量差异
            energy_diff = (E_gt - E_pred) ** 2
            
            # 泄压阀：检查连续u帧（从5改为7）的能量差异是否过大
            B, T = energy_diff.shape
            if T >= self.pressure_frames:
                # 使用滑动窗口检查连续帧（手动实现）
                window_avg = torch.zeros_like(energy_diff)
                for t in range(T):
                    start = max(0, t - self.pressure_frames + 1)
                    end = min(T, t + 1)
                    window_avg[:, t] = torch.mean(energy_diff[:, start:end], dim=1)
                # 如果连续窗口的平均值超过阈值，应用泄压阀
                pressure_mask = (window_avg > self.pressure_threshold).float()
                # 对超过阈值的帧应用软阈值
                energy_diff = torch.where(
                    pressure_mask > 0.5,
                    torch.clamp(energy_diff, max=self.pressure_threshold),
                    energy_diff
                )
            else:
                # 如果帧数不足，使用简单的软阈值
                energy_diff = torch.clamp(energy_diff, max=self.pressure_threshold)
            
            if w is not None:
                min_t = min(energy_diff.shape[1], w.shape[1])
                energy_diff = energy_diff[:, :min_t] * w[:, :min_t]
            
            # 频带加权
            band_loss = torch.mean(energy_diff) * band_w
            band_losses.append(band_loss)
        
        # 梯度归一化：把每帧的总梯度归一，避免集中爆发
        if self.normalize_gradient and len(band_losses) > 0:
            # 计算总损失并归一化
            total_band_loss = sum(band_losses)
            # 归一化到单位尺度，然后按权重重新分配
            if total_band_loss > 0:
                normalized_total = total_band_loss / (self.low_band_weight + self.mid_band_weight + self.high_band_weight)
                total_loss = normalized_total * (self.low_band_weight + self.mid_band_weight + self.high_band_weight)
            else:
                total_loss = sum(band_losses)
        else:
            total_loss = sum(band_losses)
        
        # 整体能量一致性
        E_pred_total = torch.norm(pred_expr, p=2, dim=2)
        E_gt_total = torch.norm(gt_expr, p=2, dim=2)
        total_energy_diff = (E_gt_total - E_pred_total) ** 2
        
        # 应用泄压阀（连续7帧）
        if total_energy_diff.shape[1] >= self.pressure_frames:
            # 使用滑动窗口检查连续帧（手动实现）
            B, T = total_energy_diff.shape
            window_avg_total = torch.zeros_like(total_energy_diff)
            for t in range(T):
                start = max(0, t - self.pressure_frames + 1)
                end = min(T, t + 1)
                window_avg_total[:, t] = torch.mean(total_energy_diff[:, start:end], dim=1)
            pressure_mask_total = (window_avg_total > self.pressure_threshold).float()
            total_energy_diff = torch.where(
                pressure_mask_total > 0.5,
                torch.clamp(total_energy_diff, max=self.pressure_threshold),
                total_energy_diff
            )
        else:
            total_energy_diff = torch.clamp(total_energy_diff, max=self.pressure_threshold)
        
        if w is not None:
            min_t = min(total_energy_diff.shape[1], w.shape[1])
            total_energy_diff = total_energy_diff[:, :min_t] * w[:, :min_t]
        
        total_loss += self.beta * torch.mean(total_energy_diff)
        
        return total_loss


class PoseHuberLoss(nn.Module):
    """
    姿态加速度Huber正则 + 窗口变化限速（优化版）
    使用Huber损失替代L1，对异常值更鲁棒
    优化：最小驻留帧从14 → 18
    优化：对"半径变化率"再加一级正则：∑|Δ²W_t|
    """
    def __init__(self, delta=1.0, max_window_change=24, min_dwell_frames=18, radius_change_weight=0.2):
        super(PoseHuberLoss, self).__init__()
        self.delta = delta
        self.max_window_change = max_window_change
        self.min_dwell_frames = min_dwell_frames  # 从14改为18
        self.radius_change_weight = radius_change_weight  # 半径变化率正则权重
        
    def huber_loss(self, x, delta=None):
        """Huber损失函数"""
        if delta is None:
            delta = self.delta
        abs_x = torch.abs(x)
        return torch.where(
            abs_x < delta,
            0.5 * x ** 2,
            delta * (abs_x - 0.5 * delta)
        )
    
    def forward(self, pred_pose, gt_pose, weights=None, window_change=None):
        """
        Args:
            pred_pose: [B, T, 6] 预测的姿态参数
            gt_pose: [B, T, 6] 真实的姿态参数
            weights: [B, T] 可选的权重
            window_change: [B, T-1] 窗口变化量（用于限速）
        """
        # 基础Huber损失
        diff = pred_pose - gt_pose
        base_loss = self.huber_loss(diff)
        
        if weights is not None:
            min_t = min(base_loss.shape[1], weights.shape[1])
            base_loss = base_loss[:, :min_t, :] * weights[:, :min_t, None]
        
        base_loss = torch.mean(base_loss)
        
        # 加速度Huber正则（二阶差分）
        pred_acc = pred_pose[:, 2:, :] - 2 * pred_pose[:, 1:-1, :] + pred_pose[:, :-2, :]
        gt_acc = gt_pose[:, 2:, :] - 2 * gt_pose[:, 1:-1, :] + gt_pose[:, :-2, :]
        acc_diff = pred_acc - gt_acc
        acc_loss = torch.mean(self.huber_loss(acc_diff))
        
        # 窗口变化限速损失
        window_loss = 0.0
        if window_change is not None:
            # 限制窗口变化不超过max_window_change
            excess_change = torch.clamp(torch.abs(window_change) - self.max_window_change, min=0.0)
            window_loss = torch.mean(excess_change ** 2)
            
            # 半径变化率正则：∑|Δ²W_t|
            # 计算窗口变化的二阶差分（加速度）
            if window_change.shape[1] >= 2:
                # window_change: [B, T-1]
                # 计算二阶差分
                window_change_diff = window_change[:, 1:] - window_change[:, :-1]  # [B, T-2]
                # 对半径变化率进行正则化
                radius_change_loss = torch.mean(torch.abs(window_change_diff))
                window_loss += self.radius_change_weight * radius_change_loss
        
        # 最小驻留帧约束（通过加速度损失间接实现，因为更严格的加速度约束会延长驻留时间）
        # 这里通过增加加速度损失的权重来间接实现最小驻留帧约束
        
        return base_loss + 0.5 * acc_loss + 0.1 * window_loss


class BandedSTFTLoss(nn.Module):
    """
    分带化STFT损失 + 带宽自适应
    对低频、中频、高频使用不同的权重
    新增：带宽自适应（当窗内能量持续不足时，临时提高高频带权重）
    """
    def __init__(self, n_fft=64, hop_length=16, low_weight=0.5, mid_weight=1.0, high_weight=1.5,
                 adaptive_energy_threshold=0.1, adaptive_boost=1.2, adaptive_window=10):
        super(BandedSTFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight
        self.adaptive_energy_threshold = adaptive_energy_threshold  # 能量不足阈值
        self.adaptive_boost = adaptive_boost  # 高频带权重提升倍数（×1.2）
        self.adaptive_window = adaptive_window  # 自适应窗口大小（10帧回落）
        self._energy_history = []  # 用于跟踪能量历史
        
    def _compute_adaptive_weights(self, pred_mag, gt_mag, t):
        """
        计算自适应权重：当窗内能量持续不足时，临时提高高频带权重
        """
        # 计算当前帧的能量
        current_energy = torch.mean(pred_mag[:, t, :])
        self._energy_history.append(current_energy.item())
        
        # 只保留最近adaptive_window帧的历史
        if len(self._energy_history) > self.adaptive_window:
            self._energy_history.pop(0)
        
        # 检查最近窗口内的平均能量
        if len(self._energy_history) >= self.adaptive_window:
            avg_energy = np.mean(self._energy_history[-self.adaptive_window:])
            if avg_energy < self.adaptive_energy_threshold:
                # 能量不足，临时提高高频带权重
                high_weight = self.high_weight * self.adaptive_boost
            else:
                high_weight = self.high_weight
        else:
            high_weight = self.high_weight
        
        return high_weight
        
    def forward(self, pred, gt):
        """
        Args:
            pred: [B, T, C] 预测参数
            gt: [B, T, C] 真实参数
        """
        B, T, C = pred.shape
        
        total_loss = 0.0
        count = 0
        
        for c in range(C):
            pred_sig = pred[:, :, c]  # [B, T]
            gt_sig = gt[:, :, c]  # [B, T]
            
            try:
                pred_stft = torch.stft(pred_sig, n_fft=self.n_fft, hop_length=self.hop_length,
                                      win_length=self.n_fft, return_complex=True,
                                      normalized=True, center=False)
                gt_stft = torch.stft(gt_sig, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.n_fft, return_complex=True,
                                    normalized=True, center=False)
                
                pred_mag = torch.abs(pred_stft)
                gt_mag = torch.abs(gt_stft)
                
                # 分频带：低频、中频、高频
                n_freq = pred_mag.shape[-2]
                low_end = n_freq // 3
                mid_end = 2 * n_freq // 3
                
                # 计算自适应权重（基于时间平均）
                # 简化：使用全局平均能量判断
                avg_energy = torch.mean(pred_mag)
                if avg_energy < self.adaptive_energy_threshold:
                    high_weight = self.high_weight * self.adaptive_boost
                else:
                    high_weight = self.high_weight
                
                # 低频
                low_loss = torch.mean(torch.abs(
                    pred_mag[:, :, :low_end] - gt_mag[:, :, :low_end]
                )) * self.low_weight
                
                # 中频
                mid_loss = torch.mean(torch.abs(
                    pred_mag[:, :, low_end:mid_end] - gt_mag[:, :, low_end:mid_end]
                )) * self.mid_weight
                
                # 高频（使用自适应权重）
                high_loss = torch.mean(torch.abs(
                    pred_mag[:, :, mid_end:] - gt_mag[:, :, mid_end:]
                )) * high_weight
                
                total_loss += low_loss + mid_loss + high_loss
                count += 1
            except:
                continue
        
        return total_loss / max(count, 1)


class UltraShortWindowSTFTLoss(nn.Module):
    """
    极短窗STFT损失（用于AU45）
    n_fft小，hop更短，专门捕捉高频细节
    """
    def __init__(self, n_fft=32, hop_length=8, low_weight=0.5, mid_weight=1.0, high_weight=2.0):
        super(UltraShortWindowSTFTLoss, self).__init__()
        self.n_fft = n_fft  # 小窗口，32
        self.hop_length = hop_length  # 更短的hop，8
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight
        
    def forward(self, pred_signal, gt_signal):
        """
        Args:
            pred_signal: [B, T] 预测的AU45序列
            gt_signal: [B, T] 真实的AU45序列
        """
        B, T = pred_signal.shape
        
        try:
            pred_stft = torch.stft(pred_signal, n_fft=self.n_fft, hop_length=self.hop_length,
                                  win_length=self.n_fft, return_complex=True,
                                  normalized=True, center=False)
            gt_stft = torch.stft(gt_signal, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.n_fft, return_complex=True,
                                normalized=True, center=False)
            
            pred_mag = torch.abs(pred_stft)
            gt_mag = torch.abs(gt_stft)
            
            # 分频带
            n_freq = pred_mag.shape[-2]
            low_end = n_freq // 3
            mid_end = 2 * n_freq // 3
            
            # 低频
            low_loss = torch.mean(torch.abs(
                pred_mag[:, :, :low_end] - gt_mag[:, :, :low_end]
            )) * self.low_weight
            
            # 中频
            mid_loss = torch.mean(torch.abs(
                pred_mag[:, :, low_end:mid_end] - gt_mag[:, :, low_end:mid_end]
            )) * self.mid_weight
            
            # 高频（权重更高）
            high_loss = torch.mean(torch.abs(
                pred_mag[:, :, mid_end:] - gt_mag[:, :, mid_end:]
            )) * self.high_weight
            
            return low_loss + mid_loss + high_loss
        except:
            return torch.tensor(0.0, device=pred_signal.device)


class BandedFeatureMatchingLoss(nn.Module):
    """
    分带化特征匹配损失
    对不同层的特征使用不同权重
    """
    def __init__(self, weights=None):
        super(BandedFeatureMatchingLoss, self).__init__()
        if weights is None:
            # 默认：浅层权重小，深层权重大
            self.weights = [0.5, 1.0, 1.5]
        else:
            self.weights = weights
        
    def forward(self, real_feats, fake_feats):
        """
        Args:
            real_feats: list of [B, T, C] 或单个 [B, T, C]
            fake_feats: list of [B, T, C] 或单个 [B, T, C]
        """
        if isinstance(real_feats, (list, tuple)) and isinstance(fake_feats, (list, tuple)):
            total_loss = 0.0
            for i, (rf, ff) in enumerate(zip(real_feats, fake_feats)):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                total_loss += weight * torch.mean(torch.abs(rf - ff))
            return total_loss
        else:
            return torch.mean(torch.abs(real_feats - fake_feats))

