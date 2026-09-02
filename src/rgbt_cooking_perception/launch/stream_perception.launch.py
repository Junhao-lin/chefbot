"""
ROS2 Launch 檔：啟動 RGBT 多模態串流處理與即時鍋具 Masking 節點
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rgbt_cooking_perception',
            executable='stream_processor',
            name='rgbt_stream_processor',
            output='screen',
            parameters=[{
                'sub_topic': '/rgbt/rgbt/compressed',
                'model_path': 'runs/segment/cooking_seg_3class_expert/weights/best.pt',
                'enable_yolo': True
            }]
        )
    ])
