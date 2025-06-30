#!/usr/bin/env python3
"""
临时修补Argoverse 2 LaneMarkType UNKNOWN值问题
"""

import av2.map.lane_segment as lane_segment_module
from av2.map.lane_segment import LaneMarkType
from enum import Enum

# 检查是否已经有UNKNOWN类型
if not hasattr(LaneMarkType, 'UNKNOWN'):
    print("Adding UNKNOWN to LaneMarkType...")
    
    # 创建新的LaneMarkType枚举，包含UNKNOWN
    class ExtendedLaneMarkType(Enum):
        DASH_SOLID_YELLOW = "DASH_SOLID_YELLOW"
        DASH_SOLID_WHITE = "DASH_SOLID_WHITE"
        DASHED_WHITE = "DASHED_WHITE" 
        DASHED_YELLOW = "DASHED_YELLOW"
        DOUBLE_SOLID_YELLOW = "DOUBLE_SOLID_YELLOW"
        DOUBLE_SOLID_WHITE = "DOUBLE_SOLID_WHITE"
        DOUBLE_DASH_YELLOW = "DOUBLE_DASH_YELLOW"
        DOUBLE_DASH_WHITE = "DOUBLE_DASH_WHITE"
        SOLID_YELLOW = "SOLID_YELLOW"
        SOLID_WHITE = "SOLID_WHITE"
        SOLID_DASH_WHITE = "SOLID_DASH_WHITE"
        SOLID_DASH_YELLOW = "SOLID_DASH_YELLOW"
        SOLID_BLUE = "SOLID_BLUE"
        NONE = "NONE"
        UNKNOWN = "UNKNOWN"  # 添加UNKNOWN类型

    # 替换原来的LaneMarkType
    lane_segment_module.LaneMarkType = ExtendedLaneMarkType
    
    print("✅ Successfully patched LaneMarkType!")
else:
    print("✅ LaneMarkType already has UNKNOWN") 