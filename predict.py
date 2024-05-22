# from ultralytics import YOLO
#
# # Load a pretrained YOLOv8n model
# model = YOLO('D:\\py\\cxr\\ultralytics-main\\runs\\detect\\train7\\weights\\best.pt')
#
# # Run inference on an image
# results = model('D:\\py\\yolov8\\yolov8-all-in-one\\test_data')  # list of 1 Results object
# results = model(['D:\\py\\yolov8\\yolov8-all-in-one\\test_data'])  # list of 2 Results objects

from ultralytics import YOLO

model = YOLO('D:\\py\\cxr\\ultralytics-main\\runs\\detect\\train7\\weights\\best.pt')
model.predict(source='D:\\py\\yolov8\\yolov8-all-in-one\\test_data', save=True, save_conf=True, save_txt=True, name='output')

# source后为要预测的图片数据集的的路径
# save=True为保存预测结果
# save_conf=True为保存坐标信息
# save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件