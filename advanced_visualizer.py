# advanced_visualizer.py - 高级多场景可视化
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from pathlib import Path
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder
import pandas as pd

class AdvancedQCNetVisualizer:
    def __init__(self, model_path, data_root, device='cuda:0'):
        self.device = device
        self.model = QCNet.load_from_checkpoint(model_path, map_location=device)
        self.model.eval()
        
        self.dataset = ArgoverseV2Dataset(
            root=data_root, 
            split='val',
            transform=TargetBuilder(self.model.num_historical_steps, self.model.num_future_steps)
        )
        
        # 设置绘图样式
        plt.style.use('dark_background')
        self.colors = {
            'road': '#2C2C2C',
            'lane_line': '#FFD700',
            'history': '#00BFFF',
            'future': '#32CD32',
            'prediction': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
            'agent': '#FFFFFF',
            'background': '#1C1C1C',
            'highlight': '#FF8C42'
        }
    
    def create_multi_scene_visualization(self, num_scenes=4, save_path=None):
        """创建多场景可视化，类似你提供的图片"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=self.colors['background'])
        
        for i, ax in enumerate(axes.flat):
            if i < min(num_scenes, len(self.dataset)):
                self._visualize_single_scene(ax, i, scene_id=i)
            
        plt.tight_layout(pad=2.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
            print(f"多场景可视化已保存到: {save_path}")
            
        plt.show()
    
    def _visualize_single_scene(self, ax, idx, scene_id):
        """可视化单个场景"""
        data = self.dataset[idx]
        
        # 设置背景
        ax.set_facecolor(self.colors['background'])
        
        # 获取预测结果
        with torch.no_grad():
            data_batch = data.to(self.device)
            pred = self.model(data_batch.unsqueeze(0))
        
        # 提取数据
        agent_pos = data['agent']['position'].cpu().numpy()
        map_points = data['map_point']['position'].cpu().numpy() if 'map_point' in data else None
        map_polygons = data['map_polygon']['position'].cpu().numpy() if 'map_polygon' in data else None
        
        # 绘制地图元素
        self._draw_map_elements(ax, map_points, map_polygons)
        
        # 绘制智能体轨迹
        self._draw_agent_trajectories(ax, data, pred)
        
        # 添加高亮区域
        self._add_highlight_regions(ax, agent_pos)
        
        # 设置坐标轴
        self._setup_axes(ax, agent_pos, title=f"场景 {scene_id + 1}")
    
    def _draw_map_elements(self, ax, map_points, map_polygons):
        """绘制地图元素（道路、车道线等）"""
        if map_polygons is not None:
            # 绘制道路多边形
            for i in range(0, len(map_polygons), 4):  # 假设每4个点组成一个多边形
                if i + 3 < len(map_polygons):
                    polygon_points = map_polygons[i:i+4]
                    polygon = patches.Polygon(polygon_points[:, :2], 
                                            closed=True, 
                                            facecolor=self.colors['road'], 
                                            edgecolor=self.colors['lane_line'],
                                            linewidth=0.5,
                                            alpha=0.8)
                    ax.add_patch(polygon)
        
        if map_points is not None:
            # 绘制车道线
            ax.plot(map_points[:, 0], map_points[:, 1], 
                   color=self.colors['lane_line'], 
                   linewidth=1, 
                   alpha=0.6,
                   linestyle='--')
    
    def _draw_agent_trajectories(self, ax, data, pred):
        """绘制智能体轨迹和预测"""
        # 历史轨迹
        hist_pos = data['agent']['position'][:self.model.num_historical_steps].cpu().numpy()
        ax.plot(hist_pos[:, 0], hist_pos[:, 1], 
               color=self.colors['history'], 
               linewidth=3, 
               marker='o', 
               markersize=4,
               label='历史轨迹')
        
        # 真实未来轨迹
        if 'target' in data['agent']:
            future_pos = data['agent']['target'].cpu().numpy()
            ax.plot(future_pos[:, 0], future_pos[:, 1], 
                   color=self.colors['future'], 
                   linewidth=3, 
                   marker='s', 
                   markersize=4,
                   label='真实轨迹')
        
        # 预测轨迹
        pred_pos = pred['loc_refine_pos'][0].cpu().numpy()
        probs = torch.softmax(pred['pi'][0], dim=-1).cpu().numpy()
        
        for i, (traj, prob) in enumerate(zip(pred_pos, probs)):
            color = self.colors['prediction'][i % len(self.colors['prediction'])]
            ax.plot(traj[:, 0], traj[:, 1], 
                   color=color, 
                   linewidth=2, 
                   linestyle='--',
                   alpha=0.8,
                   label=f'预测{i+1} ({prob:.2f})')
        
        # 当前位置
        current_pos = hist_pos[-1]
        ax.scatter(current_pos[0], current_pos[1], 
                  color=self.colors['agent'], 
                  s=150, 
                  marker='o', 
                  edgecolor='black',
                  linewidth=2,
                  zorder=10)
    
    def _add_highlight_regions(self, ax, agent_pos):
        """添加高亮区域（类似图片中的橙色框）"""
        # 计算智能体轨迹的边界
        x_min, x_max = agent_pos[:, 0].min(), agent_pos[:, 0].max()
        y_min, y_max = agent_pos[:, 1].min(), agent_pos[:, 1].max()
        
        # 扩展边界
        margin = 20
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin
        
        # 添加高亮矩形框
        highlight_rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3, 
            edgecolor=self.colors['highlight'], 
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(highlight_rect)
    
    def _setup_axes(self, ax, agent_pos, title):
        """设置坐标轴"""
        # 计算合适的显示范围
        center_x, center_y = agent_pos.mean(axis=0)[:2]
        range_size = 80
        
        ax.set_xlim(center_x - range_size, center_x + range_size)
        ax.set_ylim(center_y - range_size, center_y + range_size)
        ax.set_aspect('equal')
        
        # 设置样式
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
    
    def create_interactive_visualization(self, save_path=None):
        """创建交互式可视化"""
        from matplotlib.widgets import Button, Slider
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.colors['background'])
        
        # 初始场景
        self.current_scene = 0
        self._visualize_single_scene(ax, self.current_scene, self.current_scene)
        
        # 添加控制按钮
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.05])
        
        button_prev = Button(ax_prev, '上一个')
        button_next = Button(ax_next, '下一个')
        
        def next_scene(event):
            self.current_scene = (self.current_scene + 1) % min(10, len(self.dataset))
            ax.clear()
            self._visualize_single_scene(ax, self.current_scene, self.current_scene)
            plt.draw()
        
        def prev_scene(event):
            self.current_scene = (self.current_scene - 1) % min(10, len(self.dataset))
            ax.clear()
            self._visualize_single_scene(ax, self.current_scene, self.current_scene)
            plt.draw()
        
        button_next.on_clicked(next_scene)
        button_prev.on_clicked(prev_scene)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        
        plt.show()
    
    def create_video_visualization(self, output_path='qcnet_visualization.mp4', num_frames=20):
        """创建视频可视化"""
        from matplotlib.animation import FFMpegWriter
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=self.colors['background'])
        
        def animate(frame):
            for i, ax in enumerate(axes.flat):
                ax.clear()
                scene_idx = (frame + i) % min(num_frames, len(self.dataset))
                self._visualize_single_scene(ax, scene_idx, scene_idx)
        
        # 创建动画
        writer = FFMpegWriter(fps=2, metadata=dict(artist='QCNet'), bitrate=1800)
        
        with writer.saving(fig, output_path, 100):
            for frame in range(num_frames):
                animate(frame)
                writer.grab_frame()
        
        print(f"视频已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    model_path = "lightning_logs/version_0/checkpoints/epoch-0.ckpt"
    data_root = "~/test_data/argoverse_v2/"
    
    if Path(model_path).exists():
        visualizer = AdvancedQCNetVisualizer(model_path, data_root)
        
        # 创建多场景可视化（类似你的图片）
        visualizer.create_multi_scene_visualization(
            num_scenes=4, 
            save_path="advanced_multi_scene.png"
        )
        
        # 创建交互式可视化
        # visualizer.create_interactive_visualization()
        
        # 创建视频可视化
        # visualizer.create_video_visualization()
    else:
        print("请先训练模型或提供有效的模型路径") 