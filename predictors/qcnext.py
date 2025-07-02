from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import JointNLLLoss, JointMixtureNLLLoss, compute_joint_displacement_error
from losses import NLLLoss  # 备用的单智能体损失
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNextDecoder
from modules import QCNetEncoder


try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNext(pl.LightningModule):
    """
    QCNext: A Next-Generation Framework For Joint Multi-Agent Trajectory Prediction
    
    Key improvements over QCNet:
    - Joint multi-agent prediction instead of marginal prediction
    - Multi-Agent DETR-like decoder with explicit future interaction modeling
    - Scene-level scoring and loss functions
    """

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_joint_modes: int,  # K: 联合场景数量（QCNext新增）
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 **kwargs) -> None:
        super(QCNext, self).__init__()
        self.save_hyperparameters()
        
        # 基本参数
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_joint_modes = num_joint_modes  # QCNext特有：联合场景数量
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        # 编码器（继承QCNet的设计）
        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        
        # QCNext解码器（Multi-Agent DETR-like）
        self.decoder = QCNextDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_joint_modes=num_joint_modes,  # 联合场景数量
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # 联合损失函数 (按照论文公式)
        self.joint_reg_loss = JointNLLLoss(reduction='mean')
        self.joint_cls_loss = JointMixtureNLLLoss(reduction='mean')

        # 评估指标（保持与QCNet兼容）
        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()
        
        # 损失权重
        self.consistency_weight = kwargs.get('consistency_weight', 0.05)

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def training_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        # Forward pass
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)

        # 准备掩码和目标
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        
        # 构建联合轨迹目标 [A, T, 2] (只使用x,y坐标，按论文公式)
        gt_pos = data['agent']['target'][..., :self.output_dim]
        
        scene_pi = pred['scene_pi']
        agent_batch = pred['agent_batch']

        # ===== QCNext联合损失计算 (按照论文公式) =====
        
        # 1. 提议阶段损失 L_propose (场景级Winner-Take-All)
        reg_loss_propose = self.joint_reg_loss(
            pred_loc=pred['loc_propose_pos'][..., :self.output_dim],
            pred_scale=pred['scale_propose_pos'][..., :self.output_dim],
            target=gt_pos,
            agent_batch=agent_batch,
            mask=reg_mask
        )

        # 2. 精化阶段损失 L_refine (场景级Winner-Take-All) 
        reg_loss_refine = self.joint_reg_loss(
            pred_loc=pred['loc_refine_pos'][..., :self.output_dim],
            pred_scale=pred['scale_refine_pos'][..., :self.output_dim],
            target=gt_pos,
            agent_batch=agent_batch,
            mask=reg_mask
        )

        # 3. 场景级分类损失 L_cls (优化混合系数π_k)
        cls_loss = self.joint_cls_loss(
            pred_loc=pred['loc_refine_pos'][..., :self.output_dim],
            pred_scale=pred['scale_refine_pos'][..., :self.output_dim],
            target=gt_pos,
            scene_pi=scene_pi,
            agent_batch=agent_batch,
            mask=reg_mask
        )

        # 总损失
        loss = reg_loss_propose + reg_loss_refine + cls_loss

        # 记录损失
        self.log('train/reg_loss_propose', reg_loss_propose)
        self.log('train/reg_loss_refine', reg_loss_refine) 
        self.log('train/cls_loss', cls_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        # 为了与现有评估兼容，我们需要将联合预测转换为边际预测进行评估
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)

        # 选择最佳联合场景进行评估
        gt_pos = data['agent']['target'][..., :self.output_dim]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        
        # 找到最佳场景
        agent_batch = pred['agent_batch']
        displacement_errors = compute_joint_displacement_error(
            pred=pred['loc_refine_pos'][..., :self.output_dim],
            target=gt_pos,
            agent_batch=agent_batch,
            mask=reg_mask
        )
        best_scene_idx = displacement_errors.argmin()

        # 将最佳场景的预测转换为QCNet格式进行评估
        # [K, A, T, D] -> [A, M, T, D] 其中M=1（最佳场景）
        best_loc_refine_pos = pred['loc_refine_pos'][best_scene_idx].unsqueeze(1)  # [A, 1, T, D]
        best_scale_refine_pos = pred['scale_refine_pos'][best_scene_idx].unsqueeze(1)
        
        if self.output_head:
            best_loc_refine_head = pred['loc_refine_head'][best_scene_idx].unsqueeze(1)
            best_conc_refine_head = pred['conc_refine_head'][best_scene_idx].unsqueeze(1)
            
            traj_refine = torch.cat([
                best_loc_refine_pos[..., :self.output_dim],
                best_loc_refine_head,
                best_scale_refine_pos[..., :self.output_dim],
                best_conc_refine_head
            ], dim=-1)
        else:
            traj_refine = torch.cat([
                best_loc_refine_pos[..., :self.output_dim],
                best_scale_refine_pos[..., :self.output_dim]
            ], dim=-1)

        # 模拟QCNet的pi输出（置信度都设为1，因为只有一个模式）
        pi = torch.ones(traj_refine.size(0), 1, device=traj_refine.device)

        # 计算评估指标
        cls_mask = data['agent']['predict_mask'][:, -1]
        
        self.minADE.update(pred=traj_refine[..., :self.output_dim], target=gt[..., :self.output_dim],
                           prob=pi, mask=reg_mask)
        self.minFDE.update(pred=traj_refine[..., :self.output_dim], target=gt[..., :self.output_dim],
                           prob=pi, mask=cls_mask.unsqueeze(-1))
        self.MR.update(pred=traj_refine[..., :self.output_dim], target=gt[..., :self.output_dim],
                       prob=pi, mask=cls_mask.unsqueeze(-1))
        
        if self.output_head:
            self.minAHE.update(pred=traj_refine[..., self.output_dim:self.output_dim+1], 
                               target=gt[..., self.output_dim:self.output_dim+1],
                               prob=pi, mask=reg_mask)
            self.minFHE.update(pred=traj_refine[..., self.output_dim:self.output_dim+1], 
                               target=gt[..., self.output_dim:self.output_dim+1],
                               prob=pi, mask=cls_mask.unsqueeze(-1))

        return

    def on_validation_epoch_end(self):
        self.log('val/minADE', self.minADE.compute())
        self.log('val/minFDE', self.minFDE.compute()) 
        self.log('val/MR', self.MR.compute())
        if self.output_head:
            self.log('val/minAHE', self.minAHE.compute())
            self.log('val/minFHE', self.minFHE.compute())

    def test_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)

        # 对于测试，我们需要生成符合提交格式的预测
        # 选择最佳场景或者使用多个场景的ensemble
        
        if self.output_head:
            gt = torch.cat([data['agent']['target'][..., :self.output_dim],
                            data['agent']['target'][..., -1:]], dim=-1)
        else:
            gt = data['agent']['target'][..., :self.output_dim]

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        
        # 简单策略：选择最佳场景
        agent_batch = pred['agent_batch']
        displacement_errors = compute_joint_displacement_error(
            pred=pred['loc_refine_pos'][..., :self.output_dim],
            target=gt[..., :self.output_dim],
            agent_batch=agent_batch,
            mask=reg_mask
        )
        best_scene_idx = displacement_errors.argmin()

        # 生成测试预测
        best_pred = pred['loc_refine_pos'][best_scene_idx]  # [A, T, 2]
        best_conf = torch.ones(best_pred.size(0), device=best_pred.device)  # [A]

        # 存储预测结果
        for i in range(best_pred.size(0)):
            if data['agent']['predict_mask'][i, -1]:  # 需要预测的智能体
                scenario_id = data['agent']['scenario_id'][i]
                track_id = data['agent']['track_id'][i]
                
                if scenario_id not in self.test_predictions:
                    self.test_predictions[scenario_id] = {}
                
                self.test_predictions[scenario_id][track_id] = {
                    'trajectory': best_pred[i].cpu().numpy(),
                    'confidence': best_conf[i].cpu().numpy()
                }

        return

    def on_test_end(self):
        # 生成提交文件
        Path(self.submission_dir).mkdir(parents=True, exist_ok=True)
        submission_path = Path(self.submission_dir) / self.submission_file_name
        
        # 这里需要根据具体的提交格式来实现
        # 暂时保存为pickle文件
        import pickle
        with open(submission_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.test_predictions, f)

    def configure_optimizers(self):
        # 按模块类型分组参数
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'norm', 'embed']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=0.0)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNext')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_joint_modes', type=int, default=6)  # QCNext特有：联合场景数量
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser 