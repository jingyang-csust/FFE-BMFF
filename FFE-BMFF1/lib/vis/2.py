import visdom
import cv2
import torch
from lib.vis.plotting import overlay_mask, show_image_with_boxes  # 可能需要根据你的项目结构调整路径
from lib.vis.utils import numpy_to_torch  # 同样可能需要调整路径
from lib.vis.visdom_cus import Visdom

# 在需要的地方设置Visdom服务器，例如：
vis = Visdom()
# 在需要的地方设置Visdom服务器，例如：
# 示例注册一个图像数据块
image_data = torch.randn(3, 256, 256)  # 例子中的数据格式可能需要根据实际情况调整
vis.register(image_data, mode='image', title='Example Image')
# 示例更新数据
updated_image_data = torch.randn(3, 256, 256)
vis.registered_blocks['Example Image'].update(updated_image_data)
# 示例切换显示
vis.registered_blocks['Example Image'].toggle_display()
