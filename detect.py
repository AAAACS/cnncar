import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
from vgg16 import VGGnet  # 导入您自定义的 VGGnet 类

# 加载模型
model = VGGnet(num_classes=4)  # 确保num_classes与训练时一致
try:
    model.load_state_dict(torch.load('best.pt'))  # 加载权重
    print('Model weights loaded successfully.')
except Exception as e:
    print(f'Error loading model weights: {e}')

model.eval()  # 设置为评估模式

def predict_image(image_path):
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # 进行推理
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

# 示例：识别目录中的所有图像并保存结果
if __name__ == '__main__':
    image_dir = 'D:\\cnn\\carcvusts-main\\data\\images\\val'
    results = []
    total_time = 0
    num_images = 0

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            start_time = time.time()
            result = predict_image(image_path)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_images += 1
            results.append(f'{image_name}: Predicted class: {result}')

    # 计算平均推理时间
    if num_images > 0:
        avg_inference_time = total_time / num_images
        print(f'Average inference time per image: {avg_inference_time:.4f} seconds')

    # 将所有结果保存到 result.txt
    with open('result.txt', 'w') as f:
        f.write('\n'.join(results) + '\n')
        print(f'结果已保存')