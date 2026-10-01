import torch
import torch.nn as nn
from model.trans_hypothesis import FSAM
from model.extractor_dino import ViTExtractor
from torchvision import transforms

# class RGBEmbedding(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         REPO_DIR = "/home/jianliu/Diff9D/dinov3"
#         weights = "/home/jianliu/Diff9D/model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#         self.extractor = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights = weights)
#         self.extractor_preprocess = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)

#     def forward(self, rgb_raw):
#         # rgb_raw = rgb_raw.permute(0,3,1,2)
#         # rgb_raw = rgb_raw.permute(0,2,3,1)
#         rgb_raw = self.extractor_preprocess(rgb_raw)
#         #import pdb;pdb.set_trace()
#         with torch.no_grad():
#             dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:,5:]
#         # dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1)
#         return dino_feature.contiguous() # b x c x h x w
    
#     # def forward(self, inputs):
#     #     rgb=inputs['rgb']
#     #     b,rgb_h,rgb_w,_ = inputs['rgb_raw'].shape
#     #     rgb_raw = inputs['rgb_raw']
#     #     feature = self.extract_feature(rgb_raw).reshape(b,(self.num_patches)**2,-1)
#     #     return feature

# class PositionalEmbedding(torch.nn.Module):
#     def __init__(self, num_channels, max_positions=10000, endpoint=False):
#         super().__init__()
#         self.num_channels = num_channels
#         self.max_positions = max_positions
#         self.endpoint = endpoint

#     def forward(self, x):
#         freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
#         freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
#         freqs = (1 / self.max_positions) ** freqs
#         x = x.ger(freqs.to(x.dtype))
#         x = torch.cat([x.cos(), x.sin()], dim=1)
#         return x

class ShapeEstimator(nn.Module):
    def __init__(self):
        super(ShapeEstimator, self).__init__()

        self.shape_estimator = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1),
        )
        self.nocs_shape_estimator = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, rgb_global, pts_global_local):
        dim = pts_global_local.shape[2] #1024
        rgb_pts_feat = torch.cat([rgb_global.unsqueeze(2).repeat(1, 1, dim), pts_global_local], dim=1) # bs*(512+576)*1024
        shape = self.shape_estimator(rgb_pts_feat) # bs*3*1024
        nocs_shape = self.nocs_shape_estimator(rgb_pts_feat) # bs*3*1024
        return shape, nocs_shape

class shape_mlp(nn.Module):
    def __init__(self):
        super(shape_mlp, self).__init__()

        self.shape_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 192, 1),
        )
        self.nocs_shape_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 192, 1),
        )

    def forward(self, shape, nocs_shape):
        shape = self.shape_encoder(shape) # bs*192*1024
        nocs_shape = self.nocs_shape_encoder(nocs_shape) # bs*192*1024
        shape_global = torch.max(shape, dim=2, keepdim=False).values # bs*192
        nocs_shape_global = torch.max(nocs_shape, dim=2, keepdim=False).values # bs*192
        return shape_global, nocs_shape_global
   
class pose_s_condition_FCAM1(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM1, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb
    
class pose_s_condition_FCAM2(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM2, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_FCAM3(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM3, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_FCAM4(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM4, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_FCAM5(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM5, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_FCAM6(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM6, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_FCAM7(nn.Module):
    def __init__(self):
        super(pose_s_condition_FCAM7, self).__init__()
        self.denoise_net_FCAM12 = FSAM(1408, 16, True, None, 0.0, 0.0)
    def forward(self, pose_emb_init):
        pose_emb = self.denoise_net_FCAM12(pose_emb_init)
        return pose_emb

class pose_s_condition_concat2(nn.Module):
    def __init__(self):
        super(pose_s_condition_concat2, self).__init__()
        
        # self.conv_layer = nn.Conv1d(in_channels=1792*2, out_channels=1792, kernel_size=1, stride=1)
        self.concat = nn.Linear(1408*2, 1408)
        
    def forward(self, pose_emb_1, pose_emb_2):
        pose_12 = torch.cat([pose_emb_1, pose_emb_2], dim=-1) # b*1*(1792*2)
        pose_12 = self.concat(pose_12.squeeze(1))
        pose_12 = pose_12.unsqueeze(1)

        return pose_12
    
class pose_s_condition_concat1(nn.Module):
    def __init__(self):
        super(pose_s_condition_concat1, self).__init__()
        
        # self.conv_layer = nn.Conv1d(in_channels=1792*2, out_channels=1792, kernel_size=1, stride=1)
        self.concat = nn.Linear(1408*2, 1408)
        
    def forward(self, pose_emb_1, pose_emb_2):
        pose_12 = torch.cat([pose_emb_1, pose_emb_2], dim=-1) # b*1*(1792*2)
        pose_12 = self.concat(pose_12.squeeze(1))
        pose_12 = pose_12.unsqueeze(1)

        return pose_12

class pose_s_condition_concat0(nn.Module):
    def __init__(self):
        super(pose_s_condition_concat0, self).__init__()
        
        # self.conv_layer = nn.Conv1d(in_channels=1792*2, out_channels=1792, kernel_size=1, stride=1)
        self.concat = nn.Linear(1408*2, 1408)
        
    def forward(self, pose_emb_1, pose_emb_2):
        pose_12 = torch.cat([pose_emb_1, pose_emb_2], dim=-1) # b*1*(1792*2)
        pose_12 = self.concat(pose_12.squeeze(1))
        pose_12 = pose_12.unsqueeze(1)

        return pose_12


class pose_s_decoder(nn.Module):
    def __init__(self):
        super(pose_s_decoder, self).__init__()
        self.size_estimator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )    
    def forward(self, pose_emb):
        s = self.size_estimator(pose_emb.squeeze(1))
        return s  

class pose_t_decoder(nn.Module):
    def __init__(self):
        super(pose_t_decoder, self).__init__()
        self.translation_estimator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )    
    def forward(self, pose_emb):
        t = self.translation_estimator(pose_emb.squeeze(1))
        return t  

class pose_R_decoder(nn.Module):
    def __init__(self):
        super(pose_R_decoder, self).__init__()
        self.rotation_estimator = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )
        # self.rotationy_estimator = nn.Sequential(
        #     nn.Linear(1792, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3),
        # )        
    def forward(self, pose_emb):
        R = self.rotation_estimator(pose_emb.squeeze(1))
        # Ry = self.rotationy_estimator(pose_emb.squeeze(1))
        return R

class SemanticDistillationModule(nn.Module):
    def __init__(self, in_dim=512, text_dim=512):
        super(SemanticDistillationModule, self).__init__()
        # Invariant Head: 提取类内不变特征 (feat1)
        self.mlp_feat1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        # Decoder: 输出用于对齐的文本特征 (feat2)
        self.decoder_feat2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, text_dim)
        )
        # 可学习的增强系数
        # self.alpha = nn.Parameter(torch.tensor(0.1))
        self.alpha = 1

    def forward(self, x):
        # 1. 提取类内不变特征 (feat1)
        feat1 = self.mlp_feat1(x)
        # 2. 解码得到预测文本特征 (feat2)
        feat2 = self.decoder_feat2(feat1)
        # 3. 语义增强 (Add 操作)
        x_enhanced = x + self.alpha * feat1
        return x_enhanced, feat2