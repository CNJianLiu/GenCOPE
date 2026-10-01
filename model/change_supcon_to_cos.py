import torch
import clip
import torch.nn as nn
from rotation_utils import Ortho6d2Mat
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from tools.visual_points import visual_points
# from model.network import ShapeEstimator, shape_mlp
# from model.network import pose_s_condition_FCAM1, pose_s_condition_FCAM2, pose_s_condition_FCAM3, pose_s_condition_FCAM4, pose_s_condition_FCAM5, pose_s_condition_FCAM6, pose_s_condition_FCAM7, pose_s_condition_concat0, pose_s_condition_concat1, pose_s_condition_concat2
from model.network import pose_s_decoder, pose_t_decoder, pose_R_decoder, SemanticDistillationModule
from pointnet import PointNetfeat
from losses import SmoothL1Dis, ChamferDis, PoseDis
from model.transformer import ParallelTransformer
import torch.nn.functional as F
import numpy as np

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al.(http://arxiv.org/abs/1812.07035)
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

class DIFFUSION(nn.Module):
    def __init__(self):
        super(DIFFUSION, self).__init__()
        self.rgb_cam_extractor = models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
        self.rgb_cam_extractor.fc = nn.Identity()
        self.pts_mlp = PointNetfeat()

        self.pose_s_decoder = pose_s_decoder()
        self.pose_t_decoder = pose_t_decoder()
        self.pose_R_decoder = pose_R_decoder()

        # self.rgb_to_test_feat = SemanticDistillationModule()
        # self.pc_to_test_feat = SemanticDistillationModule()

        self.model1 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )

        self.model2 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )

        self.model3 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )

        self.model4 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )

        self.model5 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )   

        self.model6 = ParallelTransformer(
            d_model=512,          # input feature dimension
            nhead=8,              # 多头注意力头数
            d_ffn=2048,           # FFN dimension
            nselflayer=1,         # 可调，3层self-attn
            ncrosslayer=1,        # 可调，3层cross-attn
            dropout=0.1,
            activation='relu',
            concat=None,      # concate
            device='cuda'
        )             



    def forward(self, inputs):
        end_points = {}
        rgb = inputs['rgb']     
        # plt.imshow(rgb[0].cpu().permute(1,2,0).numpy())
        # plt.show()      
        pts = inputs['pts'] # b*1024*3
        b = rgb.size(0)
        rgb_global = self.rgb_cam_extractor(rgb) # b*512
        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c
        pts_global, pts_global_local = self.pts_mlp(pts.permute(0, 2, 1)) # b*512, b*(512+64)*1024

        # rgb_global,pred_texture_feat = self.rgb_to_test_feat(rgb_global)
        # pts_global,pred_shape_feat = self.pc_to_test_feat(pts_global)
        
        if self.training:
            text_texture_feat = None
            text_shape_feat = None
            # 支持两种输入情形：
            # 1) dataset 已经返回预计算特征: inputs 包含 'text_texture_feat' 和 'text_shape_feat'（tensor 或 numpy）
            # 2) dataset 返回字符串列表: inputs 包含 'text_texture' 和 'text_shape'（list[str]），仍然使用 CLIP 编码

            if 'text_texture_feat' in inputs and 'text_shape_feat' in inputs:
                # 直接使用预计算特征（优先）
                t_feat = inputs['text_texture_feat']
                s_feat = inputs['text_shape_feat']
                # 如果是 numpy array，转换为 tensor
                if isinstance(t_feat, np.ndarray):
                    t_feat = torch.from_numpy(t_feat)
                if isinstance(s_feat, np.ndarray):
                    s_feat = torch.from_numpy(s_feat)
                # 若是空 tensor，则置为 None
                if t_feat.numel() == 0:
                    text_texture_feat = None
                else:
                    text_texture_feat = t_feat.to(rgb.device).float()
                if s_feat.numel() == 0:
                    text_shape_feat = None
                else:
                    text_shape_feat = s_feat.to(rgb.device).float()

            end_points['rgb_feat'] = rgb_global
            end_points['pts_feat'] = pts_global

            end_points['text_texture_feat'] = text_texture_feat
            end_points['text_shape_feat'] = text_shape_feat
            # end_points['pred_texture_feat'] = pred_texture_feat
            # end_points['pred_shape_feat'] = pred_shape_feat
            end_points['category_label'] = inputs['category_label']

            gt_s = inputs['gt_s']
            gt_t = inputs['gt_t']
            gt_t = gt_t - c.squeeze(1)
            gt_R = inputs['gt_R']
            
            rgb_global = rgb_global.unsqueeze(1)   # (B,1,512)
            pts_global = pts_global.unsqueeze(1)   # (B,1,512)

            fused_rgb1, fused_pts1 = self.model1(left=rgb_global, right=pts_global)  # [B, 1, 512]*2
            fused_rgb2, fused_pts2 = self.model2(left=fused_rgb1, right=fused_pts1)  # [B, 1, 512]*2
            fused_rgb3, fused_pts3 = self.model3(left=fused_rgb2, right=fused_pts2)  # [B, 1, 512]*2
            fused_rgb4, fused_pts4 = self.model4(left=fused_rgb3, right=fused_pts3)
            fused_rgb4 = fused_rgb3 + fused_rgb4
            fused_pts4 = fused_pts3 + fused_pts4
            fused_rgb5, fused_pts5 = self.model5(left=fused_rgb4, right=fused_pts4)
            fused_rgb5 = fused_rgb2 + fused_rgb5
            fused_pts5 = fused_pts2 + fused_pts5
            fused_rgb6, fused_pts6 = self.model6(left=fused_rgb5, right=fused_pts5)
            fused_rgb6 = fused_rgb1 + fused_rgb6
            fused_pts6 = fused_pts1 + fused_pts6
            fused = torch.cat([fused_rgb6, fused_pts6], dim=-1)  # [B, 1, 1024]

            # fused_rgb3 = fused_rgb2 + fused_rgb3
            # fused_pts3 = fused_pts2 + fused_pts3
            # fused_rgb4, fused_pts4 = self.model4(left=fused_rgb3, right=fused_pts3)  # [B, 1, 512]*2
            # fused_rgb4 = fused_rgb1 + fused_rgb4
            # fused_pts4 = fused_pts1 + fused_pts4

            # fused = torch.cat([fused_rgb4, fused_pts4], dim=-1)  # [B, 1, 1024]

            pred_delta_s0 = self.pose_s_decoder(fused)
            pred_delta_t0 = self.pose_t_decoder(fused)
            pred_delta_R0 = self.pose_R_decoder(fused)
            pred_delta_R0 =  pred_delta_R0.reshape(b, 3, 3)

            end_points['pred_size0'] = pred_delta_s0
            end_points['pred_translation0'] = pred_delta_t0
            end_points['pred_rotation0'] = pred_delta_R0

            end_points['gt_s'] = gt_s
            end_points['gt_t'] = gt_t
            end_points['gt_R'] = gt_R

        else:
            with torch.no_grad():
                rgb_global = rgb_global.unsqueeze(1)   # (B,1,512)
                pts_global = pts_global.unsqueeze(1)   # (B,1,512)

                fused_rgb1, fused_pts1 = self.model1(left=rgb_global, right=pts_global)  # [B, 1, 512]*2
                fused_rgb2, fused_pts2 = self.model2(left=fused_rgb1, right=fused_pts1)  # [B, 1, 512]*2
                fused_rgb3, fused_pts3 = self.model3(left=fused_rgb2, right=fused_pts2)  # [B, 1, 512]*2
                fused_rgb4, fused_pts4 = self.model4(left=fused_rgb3, right=fused_pts3)
                fused_rgb4 = fused_rgb3 + fused_rgb4
                fused_pts4 = fused_pts3 + fused_pts4
                fused_rgb5, fused_pts5 = self.model5(left=fused_rgb4, right=fused_pts4)
                fused_rgb5 = fused_rgb2 + fused_rgb5
                fused_pts5 = fused_pts2 + fused_pts5
                fused_rgb6, fused_pts6 = self.model6(left=fused_rgb5, right=fused_pts5)
                fused_rgb6 = fused_rgb1 + fused_rgb6
                fused_pts6 = fused_pts1 + fused_pts6
                fused = torch.cat([fused_rgb6, fused_pts6], dim=-1)  # [B, 1, 1024]                

                # fused_rgb1, fused_pts1 = self.model1(left=rgb_global, right=pts_global)  # [B, 1, 512]*2
                # fused_rgb2, fused_pts2 = self.model2(left=fused_rgb1, right=fused_pts1)  # [B, 1, 512]*2
                # fused_rgb3, fused_pts3 = self.model3(left=fused_rgb2, right=fused_pts2)  # [B, 1, 512]*2
                # fused_rgb3 = fused_rgb2 + fused_rgb3
                # fused_pts3 = fused_pts2 + fused_pts3
                # fused_rgb4, fused_pts4 = self.model4(left=fused_rgb3, right=fused_pts3)  # [B, 1, 512]*2
                # fused_rgb4 = fused_rgb1 + fused_rgb4
                # fused_pts4 = fused_pts1 + fused_pts4

                # fused = torch.cat([fused_rgb4, fused_pts4], dim=-1)  # [B, 1, 1024]

                pred_delta_s = self.pose_s_decoder(fused)
                pred_delta_t = self.pose_t_decoder(fused)
                pred_delta_R = self.pose_R_decoder(fused)
                pred_delta_R =  pred_delta_R.reshape(b, 3, 3)

            end_points['pred_size'] = pred_delta_s
            end_points['pred_translation'] = pred_delta_t + c.squeeze(1)
            end_points['pred_rotation'] = pred_delta_R

        return end_points


class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg=cfg.loss
        self.freeze_world_enhancer=cfg.freeze_world_enhancer
        learnable_temp = False
        temperature: float = 0.07
        self.lambda_rgb: float = 0.1
        self.lambda_pc: float = 0.1
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        else:
            self.temperature = temperature

    def forward(self, end_points):
        s0 = end_points['pred_size0']
        t0 = end_points['pred_translation0']
        R0 = end_points['pred_rotation0']

        s = end_points['gt_s']
        t = end_points['gt_t']
        R = end_points['gt_R']

        ids = end_points['category_label']

        loss_pose = self._get_loss(s0, t0, R0, s, t, R)

        rgb_feature_norm = F.normalize(end_points['rgb_feat'], dim=-1)
        pts_feature_norm = F.normalize(end_points['pts_feat'], dim=-1)
        text_texture_feature_norm = F.normalize(end_points['text_texture_feat'], dim=-1)
        text_shape_feature_norm = F.normalize(end_points['text_shape_feat'], dim=-1)
        cos_sim = nn.CosineSimilarity(dim=-1)
        loss_rgb_2d = (1 - cos_sim(rgb_feature_norm, text_texture_feature_norm).mean()) * self.lambda_rgb
        loss_pc_3d = (1 - cos_sim(pts_feature_norm, text_shape_feature_norm).mean()) * self.lambda_pc

        # loss_clip = self._clip_loss(end_points['rgb_feat'], end_points['pts_feat'], end_points['text_texture_feat'], end_points['text_shape_feat']) 

        # loss_rgb_2d = self._compute_supcon_loss(end_points['rgb_feat'], end_points['text_texture_feat'], ids) * self.lambda_rgb
        # loss_pc_3d = self._compute_supcon_loss(end_points['pts_feat'], end_points['text_shape_feat'], ids) * self.lambda_pc
        # loss_rgb_pc = self._compute_supcon_loss(end_points['rgb_feat'], end_points['pts_feat'], ids) * self.lambda_cross

        total_loss = loss_pose + loss_rgb_2d + loss_pc_3d
        # total_loss = loss_pose + loss_clip['total_loss']
        try:
            self.last_loss_dict = {
                'loss_pose': float(loss_pose.item()),
                'loss_rgb_2d': float(loss_rgb_2d.item()),
                'loss_pc_3d': float(loss_pc_3d.item()),
                # 'loss_rgb_pc': float(loss_rgb_pc.item()),
            }
        except Exception:
            self.last_loss_dict = None
        return total_loss
    
    def _get_loss(self, s0, t0, R0, s1, t1, R1):
        loss_pose = PoseDis(s0, t0, R0, s1, t1, R1)
        return loss_pose

    def _get_loss_shape(self, shape_pred, shape_truth):
        loss_shape = ChamferDis(shape_pred, shape_truth)
        return loss_shape
    
    def _get_loss_nocs_shape(self, nocs_shape_pred, nocs_shape_truth):
        loss_nocs_shape = SmoothL1Dis(nocs_shape_pred, nocs_shape_truth)
        return loss_nocs_shape

    def _compute_supcon_loss(self, features_a, features_b, labels):
        """
        计算基于类别的监督对比损失 (支持多对多正样本)
        输入:
            features_a: (Batch, Dim)
            features_b: (Batch, Dim)
            labels: (Batch, )  例如: [0, 0, 1, 5, 2...]
        """
        device = features_a.device
        batch_size = features_a.shape[0]

        # 1. 特征归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)

        # 2. 计算相似度矩阵 (B x B)
        # logits[i][j] = feat_a[i] dot feat_b[j]
        logits = torch.matmul(features_a, features_b.T) / self.temperature

        # 3. 生成正样本掩码 (Mask)
        # labels: (B, 1)
        labels = labels.contiguous().view(-1, 1)
        # mask[i][j] = 1 if label[i] == label[j] else 0
        mask = torch.eq(labels, labels.T).float().to(device)

        # 4. 计算 Logits 的 Log-Softmax
        # 为了数值稳定性，先减去每行的最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 计算分母：Sum(exp(all_logits))
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 5. 计算 Loss
        # 公式： - (1 / |P(i)|) * Sum_{p in P(i)} log_prob[i][p]
        # 只保留正样本位置的 log_prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        
        return loss        
    
    def _compute_contrastive_loss(
        self,
        features: torch.Tensor,  # [B, D]
        text_features: torch.Tensor,  # [B, D]
        logit_scale: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算对比学习损失（InfoNCE损失）
        """
        # 确保 dtype 与 device 一致（clip 返回可能为 float16）
        if text_features is not None:
            text_features = text_features.to(device=features.device, dtype=features.dtype)

        B = features.shape[0]
        D = features.shape[1]
        # 特征归一化
        features = F.normalize(features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        # features = features.unsqueeze(2)  # [B, D, 1]
        # text_features = text_features.unsqueeze(1)  # [B, 1, D]
        
        # 计算相似度矩阵
        if logit_scale is not None:
            logit_scale = logit_scale.to(features.dtype)
            logits_per_feature = logit_scale * (features.unsqueeze(2) @ text_features.unsqueeze(1)) # [B, D, D]
            logits_per_text = logit_scale * (text_features.unsqueeze(2) @ features.unsqueeze(1)) # [B, D, D]
            logits_per_feature_flat = logits_per_feature.reshape(B*D, D)
            logits_per_text_flat = logits_per_text.reshape(B*D, D)
        else:
            temperature = torch.tensor(self.temperature, device=features.device, dtype=features.dtype)
            logits_per_feature = (features.unsqueeze(2) @ text_features.unsqueeze(1)) / temperature
            logits_per_text = (text_features.unsqueeze(2) @ features.unsqueeze(1)) / temperature
            logits_per_feature_flat = logits_per_feature.reshape(B*D, D)
            logits_per_text_flat = logits_per_text.reshape(B*D, D)
        
        # 创建标签（对角线匹配）
        labels = torch.arange(D, device=features.device).repeat(B, 1)
        # labels = torch.arange(batch_size, device=features.device)
        labels_flat = labels.reshape(B*D)
        
        # 计算交叉熵损失
        loss_i = F.cross_entropy(logits_per_feature_flat, labels_flat)
        loss_t = F.cross_entropy(logits_per_text_flat, labels_flat)
        
        return (loss_i + loss_t) / 2.0
    
    def _clip_loss(
        self,
        rgb_features: torch.Tensor,          # [B, D_rgb]
        pointcloud_features: torch.Tensor,    # [B, D_pc]
        text_2d_features: torch.Tensor,       # [B, D_text] - 描述2D特征的文本
        text_3d_features: torch.Tensor,       # [B, D_text] - 描述3D特征的文本
    ) -> dict:
        """
        前向计算损失
        """
        losses = {}
        
        # 获取温度参数
        if hasattr(self, 'logit_scale'):
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = None
        
        # 1. RGB与2D文本的对比损失
        loss_rgb_2d = self._compute_contrastive_loss(
            rgb_features, text_2d_features, logit_scale
        )
        losses['loss_rgb_2d'] = loss_rgb_2d * self.lambda_rgb
        
        # 2. 点云与3D文本的对比损失
        loss_pc_3d = self._compute_contrastive_loss(
            pointcloud_features, text_3d_features, logit_scale
        )
        losses['loss_pc_3d'] = loss_pc_3d * self.lambda_pc
        
        # # 3. 跨模态对齐损失（可选但推荐）
        # if self.lambda_cross > 0:
        #     # RGB与3D文本的部分对齐（因为RGB也能反映3D信息）
        #     loss_rgb_3d = self._compute_contrastive_loss(
        #         rgb_features, text_3d_features, logit_scale
        #     )
            
        #     # 点云与2D文本的部分对齐（因为点云投影到2D）
        #     loss_pc_2d = self._compute_contrastive_loss(
        #         pointcloud_features, text_2d_features, logit_scale
        #     )
            
        #     losses['loss_cross'] = (loss_rgb_3d + loss_pc_2d) / 2 * self.lambda_cross
        
        # 4. 模态间一致性损失（如果特征维度相同）
        if rgb_features.shape[-1] == pointcloud_features.shape[-1]:
            # 鼓励RGB和点云特征在共享空间中对齐
            loss_modality_align = F.mse_loss(
                F.normalize(rgb_features, dim=-1),
                F.normalize(pointcloud_features, dim=-1)
            )
            losses['loss_modality_align'] = loss_modality_align * 0.1
        
        # 计算总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses    
    
