import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from ScreenCapture import ScreenCapture  # 导入你的ScreenCapture类


def run(sc):
    # 实例化 YOLO 模型
    model = YOLO("D:\\py\\cxr\\ultralytics-main\\runs\\detect\\train7\\weights\\best.pt")
    # 设置字体
    fontStyle = ImageFont.truetype("./font/simsun.ttc", 48, encoding="utf-8")  # 中文字体文件

    # 循环体
    while True:
        # 截屏
        img = Image.fromarray(np.uint8(sc.grab_screen_mss()))
        # 实例化一个图像绘制对象
        draw = ImageDraw.Draw(img)

        # 利用模型进行图像预测
        results = model.predict(source=img, conf=0.75, iou=0.75)

        # 遍历结果，detection
        for result in results:
            # detection
            if len(result.boxes.xyxy) > 0:
                boxes_xyxy = result.boxes.xyxy.tolist()
                boxes_cls = result.boxes.cls.tolist()

                for i, box_xyxy in enumerate(boxes_xyxy):
                    draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                    draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                              font=fontStyle)

        cv2.namedWindow(sc.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(sc.window_name, sc.display_window_width, sc.display_window_height)
        cv2.imshow(sc.window_name, np.array(img))

        if cv2.waitKey(1) & 0xFF == sc.exit_code:  # 默认：ESC
            cv2.destroyAllWindows()
            os._exit(0)

        # 短暂延迟，以便窗口有时间刷新
        cv2.waitKey(10)


if __name__ == '__main__':
    sc = ScreenCapture()
    run(sc)

