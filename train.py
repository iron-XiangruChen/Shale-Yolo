# from ultralytics import YOLO
#
# # 加载模型
# model = YOLO('D:\\py\\cxr\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8.yaml').load('D:\\py\\cxr\\ultralytics-main\\yolov8n.pt')  # 从YAML构建并转移权重
#
# if __name__ == '__main__':
#     # 训练模型
#     results = model.train(data='D:\\py\\cxr\\ultralytics-main\\ultralytics\\cfg\\datasets\\my_coco128.yaml', epochs=10, imgsz=640, batch=24)
#
#     metrics = model.val()
#
from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO('D:\\py\\cxr\\ultralytics-main\\add3_yolov8n.yaml')  # build a new model from YAML
    model = YOLO('D:\\py\\cxr\\ultralytics-main\\yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('D:\\py\\cxr\\ultralytics-main\\add3_yolov8n.yaml').load('D:\\py\\cxr\\ultralytics-main\\yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='D:\\py\\cxr\\ultralytics-main\\ultralytics\\cfg\\datasets\\my_coco128.yaml', epochs=200, imgsz=640, batch=24, workers=4)