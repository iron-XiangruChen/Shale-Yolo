import mss
import cv2
import numpy as np

class ScreenCapture:
  def __init__(self, screen_resolution=(1920, 1080), capture_region=(1, 1), window_name='test', exit_code=0x1B):

    self.screen_capture = mss.mss() # 实例化mss
    self.screen_width = screen_resolution[0] # 屏幕的宽
    self.screen_height = screen_resolution[1] # 屏幕的高
    self.capture_region = capture_region # 捕获区域
    self.screen_center_x, self.screen_center_y = self.screen_width // 2, self.screen_height // 2 # 屏幕中心点坐标
    # 截图区域
    self.capture_width, self.capture_height = int(self.screen_width * self.capture_region[0]), int(
      self.screen_height * self.capture_region[1]) # 宽高
    self.capture_left, self.capture_top = int(

      0 + self.screen_width // 2 * (1. - self.capture_region[0])), int(

      0 + self.screen_height // 2 * (1. - self.capture_region[1])) # 原点

    self.display_window_width, self.display_window_height = self.screen_width // 1, self.screen_height // 1 # 显示窗口大小

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

# 创建 ScreenCapture 实例
screen_capture = ScreenCapture()

# 捕获屏幕截图
screen_img = screen_capture.grab_screen_mss()

# 显示截图
cv2.imshow(screen_capture.window_name, screen_img)

# 检测按键事件
while True:
    if cv2.waitKey(1) & 0xFF == screen_capture.exit_code:
        break

# 释放窗口
cv2.destroyAllWindows()

# import mss
# import cv2
# import numpy as np
#
# class ScreenCapture:
#     def __init__(self, screen_resolution=(1920, 1080), capture_region=(0.5, 0.5), window_name='test', exit_code=0x1B):
#         self.screen_capture = mss.mss()  # 实例化mss
#         self.screen_width = screen_resolution[0]  # 屏幕的宽
#         self.screen_height = screen_resolution[1]  # 屏幕的高
#         self.capture_region = capture_region  # 捕获区域
#         self.screen_center_x, self.screen_center_y = self.screen_width // 2, self.screen_height // 2  # 屏幕中心点坐标
#         # 截图区域
#         self.capture_width, self.capture_height = int(self.screen_width * self.capture_region[0]), int(
#             self.screen_height * self.capture_region[1])  # 宽高
#         self.capture_left, self.capture_top = int(
#             0 + self.screen_width // 2 * (1. - self.capture_region[0])), int(
#             0 + self.screen_height // 2 * (1. - self.capture_region[1]))  # 原点
#
#         self.display_window_width, self.display_window_height = self.screen_width // 3, self.screen_height // 3  # 显示窗口大小
#
#         self.monitor_settings = {
#             'left': self.capture_left,
#             'top': self.capture_top,
#             'width': self.capture_width,
#             'height': self.capture_height
#         }
#         self.window_name = window_name
#         self.exit_code = exit_code
#         self.img = None
#
#     def grab_screen_mss(self):
#         return cv2.cvtColor(np.array(self.screen_capture.grab(self.monitor_settings)), cv2.COLOR_BGRA2BGR)
#
# # 创建 ScreenCapture 实例
# screen_capture = ScreenCapture()
#
# # 捕获屏幕截图
# screen_img = screen_capture.grab_screen_mss()
#
# # 显示截图
# cv2.namedWindow(screen_capture.window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(screen_capture.window_name, screen_capture.display_window_width, screen_capture.display_window_height)
# cv2.imshow(screen_capture.window_name, screen_img)
#
# # 检测按键事件
# while True:
#     if cv2.waitKey(1) & 0xFF == screen_capture.exit_code:
#         break
#
# # 释放窗口
# cv2.destroyAllWindows()
