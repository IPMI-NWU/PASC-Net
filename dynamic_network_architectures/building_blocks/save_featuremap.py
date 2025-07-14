from PIL import Image
import numpy as np

def save_map(x, name, channel):
    # 假设 x 是一个 PyTorch Tensor，你可以将其转换为 NumPy 数组
    x_np = x.detach().cpu().numpy()

    # 假设 x_np 的形状是 (batch_size, channels, height, width)，选择一个示例
    single_example = x_np[0]  # 选择第一个样本的特征图
    

    # 选择一个通道
    channel_to_visualize = channel
    feature_map = single_example[channel_to_visualize, :, :]
    matrix = np.array(feature_map)

    # 将矩阵的值缩放到 0-255（灰度图的像素值范围）
    matrix_scaled = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)
    print(matrix_scaled.shape)
    image = Image.fromarray(matrix_scaled, mode='L')  # 'L' 表示灰度图像模式

    # 保存图像
    image.save(name + '.png')
