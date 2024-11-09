import cv2
import numpy as np

# 假设图像大小为 800x600
image_width = 800
image_height = 600

# 目标位置参数
cx = 400  # 中心点横坐标
cy = 300  # 中心点纵坐标
w = 200   # 框的宽度
h = 150   # 框的高度

# 计算矩形框的左上角和右下角坐标
x1 = int(cx - w//2)
y1 = int(cy - h//2)
x2 = int(cx + w//2)
y2 = int(cy + h//2)

# 创建一张空白图像，或者从文件中读取图像
# 这里假设直接创建一张空白图像作为示例
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# 画出矩形框
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (0, 255, 0) 是颜色，2 是线条粗细

# 显示图像
cv2.imshow('Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
