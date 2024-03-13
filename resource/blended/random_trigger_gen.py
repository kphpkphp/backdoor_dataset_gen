from PIL import Image
import numpy as np
# 生成随机像素数据
random_pixels = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
# 创建 PIL 图像对象
image = Image.fromarray(random_pixels)
# 保存图像
image.save("random_image.png")

# 显示图像（可选）
# image.show()
