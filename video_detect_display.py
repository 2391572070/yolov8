import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 初始化YOLOv8模型

mdl = '/home/skon/PycharmProjects/yolov8-sida/runs/detect/train6/weights/best.pt'
# 设置自己训练好的模型路径
model = YOLO(mdl)
videos_root_path = '/media/skon/Data/Data/Upload_video/out'
detected_videos_path = '/media/skon/Data/Data/Upload_video'
out_branch_name = 'object_detected'

pause = True

def os_system(cmd_str):
    if sys.platform.startswith('linux'):
        os.system(cmd_str)


for filename in os.listdir(videos_root_path):
        if filename.endswith(".mp4"):
            # 读取视频文件
            # cap = cv2.VideoCapture(0) # 打开笔记本摄像头
            video_file_path = os.path.join(videos_root_path, filename)
            cap = cv2.VideoCapture(video_file_path)
            # 获取原视频的宽度和高度
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置新的视频帧大小
            new_width = 1280
            new_height = 720

            # 设置保存视频的文件名、编解码器和帧速率
            output_path = os.path.join(detected_videos_path, out_branch_name)  # 替换为你的输出视频文件路径
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(detected_videos_path, out_branch_name, filename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (original_width, original_height))


            # 逐帧进行预测
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 对每一帧进行预测。并设置置信度阈值为0.8，需要其他参数，可直接在后面加
                results = model(frame, False, conf=0.5, branchID=2, branchcls_start=12)
                conf = True
                # 绘制预测结果
                for result in results:
                    # 绘制矩形框
                    for box in result.boxes:
                        xyxy = box.xyxy.squeeze().tolist()
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        c, conf, id = int(box.cls), float(box.conf) if conf else None, None if box.id is None else int(
                            box.id.item())
                        name = ('' if id is None else f'id:{id} ') + result.names[c]
                        label = name
                        confidence = conf
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)
                # 或者使用下行代码绘制所有结果
                # res=results[0].plot(conf=False)
                os_system('chmod a+wr \"{}\"'.format(output_path))
                # 显示预测结果
                out.write(frame)
                # cv2.imshow("Predictions", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break


                # if pause:
                #     key = cv2.waitKey()
                # else:
                #     key = cv2.waitKey(1)
                # if key == 27 or key == ord('q') or key == ord('Q'):
                #     break

# 释放资源并关闭窗口
            cap.release()
            out.release()
        cv2.destroyAllWindows()
