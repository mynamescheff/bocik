import numpy as np
import cv2 as cv
import win32gui
import win32ui
import win32con
from time import time


window_title = "Ervelia"
x_corr = 8
y_corr = 31
margin = 50

class HiddenWindowCapture:
    def __init__(self, window_title):
        self.hwnd = win32gui.FindWindow(None, window_title)
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = self.window_rect[2] - self.window_rect[0] - 2 * margin
        self.h = self.window_rect[3] - self.window_rect[1] - 2 * margin
        self.cropped_x = x_corr + margin
        self.cropped_y = y_corr + margin

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]
        img = np.ascontiguousarray(img)

        return img
    
    def get_grayscale_screenshot(self):
        screenshot = self.get_screenshot()
        grayscale_screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        return grayscale_screenshot

capture = HiddenWindowCapture(window_title)

while True:
    grayscale_screenshot = capture.get_grayscale_screenshot()
    cv.imshow("Grayscale Screenshot", grayscale_screenshot)

    loop_time = time()

    key = cv.waitKey(5000)
    if key == ord("q"):
        cv.destroyAllWindows()
        break
    elif key == ord("f"):
        print("[INFO] Screenshot taken...")
        cv.imwrite("screenshots/{}_gray.jpg".format(loop_time), grayscale_screenshot)

print("[INFO] Done.")