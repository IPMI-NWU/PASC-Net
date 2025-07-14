from PIL import Image
import numpy as np

def save_map(x, name, channel):
    # ���� x ��һ�� PyTorch Tensor������Խ���ת��Ϊ NumPy ����
    x_np = x.detach().cpu().numpy()

    # ���� x_np ����״�� (batch_size, channels, height, width)��ѡ��һ��ʾ��
    single_example = x_np[0]  # ѡ���һ������������ͼ
    

    # ѡ��һ��ͨ��
    channel_to_visualize = channel
    feature_map = single_example[channel_to_visualize, :, :]
    matrix = np.array(feature_map)

    # �������ֵ���ŵ� 0-255���Ҷ�ͼ������ֵ��Χ��
    matrix_scaled = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)
    print(matrix_scaled.shape)
    image = Image.fromarray(matrix_scaled, mode='L')  # 'L' ��ʾ�Ҷ�ͼ��ģʽ

    # ����ͼ��
    image.save(name + '.png')
