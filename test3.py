from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

# 初始化人脸检测器和人脸识别模型
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 加载目标人脸图像并计算嵌入
target_image_path = 'target.jpg'
target_image = Image.open(target_image_path)
target_image_aligned, _ = mtcnn(target_image, return_prob=True)

# 检查并调整目标人脸的格式
if target_image_aligned is not None:
    target_image_aligned = target_image_aligned.to(device)
    target_embedding = resnet(target_image_aligned).detach().cpu()
else:
    raise ValueError("未检测到目标人脸，请检查目标图像")

# 加载视频文件
video_path = 'video.mp4'
video = cv2.VideoCapture(video_path)

# 设置视频输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
out = cv2.VideoWriter('tracked_video.mp4', fourcc, 25.0, (640, 360))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # 将帧转换为 PIL 图像格式并进行人脸检测
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    aligned_faces, probs = mtcnn(frame_pil, return_prob=True)

    if aligned_faces is not None:
        aligned_faces = aligned_faces.to(device)

        embeddings = resnet(aligned_faces).detach().cpu()

        # 计算目标嵌入与检测到的每张人脸之间的距离
        distances = [(target_embedding - emb).norm().item() for emb in embeddings]
        min_dist_index = np.argmin(distances)  # 找到最小距离的索引
        min_dist_box = mtcnn.detect(frame_pil)[0][min_dist_index]

        # 在帧上绘制跟踪的边界框
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle(min_dist_box.tolist(), outline=(255, 0, 0), width=6)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 显示并保存当前帧
    cv2.imshow('Face Verification and Tracking', frame)
    out.write(cv2.resize(frame, (640, 360)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video.release()
out.release()
cv2.destroyAllWindows()
