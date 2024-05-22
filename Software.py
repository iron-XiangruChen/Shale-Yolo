import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import mss
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap, QFont, QTransform
from PyQt5.QtCore import Qt


class ScreenCapture:
    def __init__(self, screen_resolution=(1920, 1080), capture_region=(1, 1), window_name='test', exit_code=0x1B):
        self.screen_capture = mss.mss()
        self.screen_width = screen_resolution[0]
        self.screen_height = screen_resolution[1]
        self.capture_region = capture_region
        self.screen_center_x, self.screen_center_y = self.screen_width // 2, self.screen_height // 2

        self.capture_width, self.capture_height = int(self.screen_width * self.capture_region[0]), int(
            self.screen_height * self.capture_region[1])
        self.capture_left, self.capture_top = int(
            0 + self.screen_width // 2 * (1. - self.capture_region[0])), int(
            0 + self.screen_height // 2 * (1. - self.capture_region[1]))

        self.display_window_width, self.display_window_height = self.screen_width // 1, self.screen_height // 1

        self.monitor_settings = {
            'left': self.capture_left,
            'top': self.capture_top,
            'width': self.capture_width,
            'height': self.capture_height
        }
        self.window_name = window_name
        self.exit_code = exit_code
        self.img = None

    def grab_screen_mss(self):
        return cv2.cvtColor(np.array(self.screen_capture.grab(self.monitor_settings)), cv2.COLOR_BGRA2BGR)


class YoloApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def importImage(self):
        if self.model is None:
            print("请先选择模型!")
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg);;All Files (*)",
                                                  options=options)

        if fileName:
            img = Image.open(fileName)
            results = self.model.predict(source=img, conf=0.75, iou=0.75)

            for result in results:
                if len(result.boxes.xyxy) > 0:
                    draw = ImageDraw.Draw(img)
                    boxes_xyxy = result.boxes.xyxy.tolist()
                    boxes_cls = result.boxes.cls.tolist()

                    for i, box_xyxy in enumerate(boxes_xyxy):
                        draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                        draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                  font=self.fontStyle)

            # 保存图像到用户选择的文件夹
            options = QFileDialog.Options()
            folderName = QFileDialog.getExistingDirectory(self, "选择保存文件夹", options=options)
            if folderName:
                img.save(folderName + "/detection_result.png")
                print(f"图片已保存到 {folderName}/detection_result.png")

            # 在显示图像窗口中显示带有检测框的图像
            self.showImage(img)

    def initUI(self):
        self.setWindowTitle('Shale-Yolo：Target detection software for shale SEM images of microscopic materials')
        self.setGeometry(100, 100, 1000, 1000)

        self.screen_capture = ScreenCapture()
        self.model = None
        self.fontStyle = ImageFont.truetype("./font/simsun.ttc", 48, encoding="utf-8")

        self.selectModelBtn = QPushButton('Model', self)
        self.selectModelBtn.clicked.connect(self.selectModel)
        self.selectModelBtn.setGeometry(50, 50, 300, 50)
        self.selectModelBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.detectScreenBtn = QPushButton('Detect Screen', self)
        self.detectScreenBtn.clicked.connect(self.detectScreen)
        self.detectScreenBtn.setGeometry(50, 120, 300, 50)
        self.detectScreenBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.importImageBtn = QPushButton('Import Image', self)
        self.importImageBtn.clicked.connect(self.importImage)
        self.importImageBtn.setGeometry(50, 190, 300, 50)
        self.importImageBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.importVideoBtn = QPushButton('Import Video', self)
        self.importVideoBtn.clicked.connect(self.importVideo)
        self.importVideoBtn.setGeometry(600, 50, 300, 50)
        self.importVideoBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.detectCameraBtn = QPushButton('Detect Camera', self)
        self.detectCameraBtn.clicked.connect(self.detectCamera)
        self.detectCameraBtn.setGeometry(600, 120, 300, 50)
        self.detectCameraBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.saveFileBtn = QPushButton('Save File', self)
        self.saveFileBtn.clicked.connect(self.saveFile)
        self.saveFileBtn.setGeometry(600, 190, 300, 50)
        self.saveFileBtn.setFont(QFont("Times New Roman", 20))  # 设置字体为Times New Roman，大小为20

        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(50, 260, 850, 600)  # 调整显示窗口大小
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("border: 2px solid black;")

    def selectModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO Model Files (*.pt);;All Files (*)",
                                                  options=options)
        if fileName:
            self.model = YOLO(fileName)

    def showImage(self, image):
        qimage = QImage(image.tobytes("raw", "RGB"), image.width, image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        # 调整图像大小以适应QLabel
        pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), Qt.KeepAspectRatio)

        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.update()

    def detectScreen(self):
        if self.model is None:
            print("请先选择模型!")
            return

        while True:
            img = Image.fromarray(np.uint8(self.screen_capture.grab_screen_mss()))
            draw = ImageDraw.Draw(img)

            results = self.model.predict(source=img, conf=0.75, iou=0.75)

            for result in results:
                if len(result.boxes.xyxy) > 0:
                    boxes_xyxy = result.boxes.xyxy.tolist()
                    boxes_cls = result.boxes.cls.tolist()

                    for i, box_xyxy in enumerate(boxes_xyxy):
                        draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                        draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                  font=self.fontStyle)

            cv2.namedWindow(self.screen_capture.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.screen_capture.window_name, self.screen_capture.display_window_width,
                             self.screen_capture.display_window_height)
            cv2.imshow(self.screen_capture.window_name, np.array(img))

            if cv2.waitKey(1) & 0xFF == self.screen_capture.exit_code:
                cv2.destroyAllWindows()
                break
                self.showImage(img)

    def importVideo(self):
        if self.model is None:
            print("请先选择模型!")
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "导入视频文件", "", "Video Files (*.mp4 *.avi);;All Files (*)",
                                                  options=options)
        if fileName:
            cap = cv2.VideoCapture(fileName)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.model.predict(source=img, conf=0.75, iou=0.75)

                for result in results:
                    if len(result.boxes.xyxy) > 0:
                        draw = ImageDraw.Draw(img)
                        boxes_xyxy = result.boxes.xyxy.tolist()
                        boxes_cls = result.boxes.cls.tolist
                    for i, box_xyxy in enumerate(boxes_xyxy):
                        draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                        draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                  font=self.fontStyle)

                cv2.imshow('Video', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            frame_with_detections = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # 在显示图像窗口中显示带有检测框的图像
            self.showImage(frame_with_detections)

    def detectCamera(self):
        if self.model is None:
            print("请先选择模型!")
            return

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            results = self.model.predict(source=img, conf=0.75, iou=0.75)

            for result in results:
                if len(result.boxes.xyxy) > 0:
                    boxes_xyxy = result.boxes.xyxy.tolist()
                    boxes_cls = result.boxes.cls.tolist()

                    for i, box_xyxy in enumerate(boxes_xyxy):
                        draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                        draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                  font=self.fontStyle)

            cv2.imshow('Camera', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        frame_with_detections = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # 在显示图像窗口中显示带有检测框的图像
        self.showImage(frame_with_detections)

    def saveFile(self):
        if self.model is None:
            print("请先选择模型!")
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "保存文件", "",
                                                  "Image Files (*.png *.jpg);;Video Files (*.mp4 *.avi);;All Files (*)",
                                                  options=options)

        if fileName:
            if fileName.lower().endswith(('.png', '.jpg')):
                img = Image.fromarray(np.uint8(self.screen_capture.grab_screen_mss()))
                results = self.model.predict(source=img, conf=0.75, iou=0.75)

                for result in results:
                    if len(result.boxes.xyxy) > 0:
                        draw = ImageDraw.Draw(img)
                        boxes_xyxy = result.boxes.xyxy.tolist()
                        boxes_cls = result.boxes.cls.tolist()

                        for i, box_xyxy in enumerate(boxes_xyxy):
                            draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                            draw.text((int(box_xyxy[0]), int(box_xyxy[1]) - 20), str(int(boxes_cls[i])), (0, 0, 255),
                                      font=self.fontStyle)

                img.save(fileName)
                print(f"图片已保存到 {fileName}")

            elif fileName.lower().endswith(('.mp4', '.avi')):
                cap = cv2.VideoCapture(0)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(fileName, fourcc, 20.0, (640, 480))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = self.model.predict(source=img, conf=0.75, iou=0.75)

                    for result in results:
                        if len(result.boxes.xyxy) > 0:
                            draw = ImageDraw.Draw(img)
                            boxes_xyxy = result.boxes.xyxy.tolist()
                            boxes_cls = result.boxes.cls.tolist()

                            for i, box_xyxy in enumerate(boxes_xyxy):
                                draw.rectangle(box_xyxy, outline=(0, 0, 255), width=5)
                                draw.text((int(box_xyxy[0]),
                                int(box_xyxy[1]) - 20), str(int(boxes_cls[i])),
                                          (0, 0, 255),
                                          font=self.fontStyle)

                    frame_with_detections = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    out.write(frame_with_detections)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()
                print(f"视频已保存到 {fileName}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YoloApp()
    window.show()
    sys.exit(app.exec_())
