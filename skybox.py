import cv2
import numpy as np
from PIL import Image
import sys
import math
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
import zipfile
import os
import shutil
from PySide6 import QtGui

class PerspectiveImagePaster:

    def __init__(self):
        return

    def paste_image_by_perspective(self, base_img, overlay_img, target_points, antialiasing=False, hard_edge=False, src_points=None):
        if len(target_points) != 4:
            raise ValueError('need 4 point required')

        dst_pts = np.array(target_points, dtype=np.float32)
        h, w = overlay_img.shape[:2]
        if src_points is None:
            src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        else:
            src_pts = np.array(src_points, dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        bh, bw = base_img.shape[:2]
        warped_img = cv2.warpPerspective(overlay_img, M, (bw, bh), 
                                         flags=cv2.INTER_LINEAR if antialiasing else cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE)

        if len(base_img.shape) == 3 and len(warped_img.shape) == 3:
            if base_img.shape[2] == 3 and warped_img.shape[2] == 4:
                warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGRA2BGR)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = 255
        warped_mask = cv2.warpPerspective(mask, M, (bw, bh), 
                                          flags=cv2.INTER_LINEAR if antialiasing else cv2.INTER_NEAREST)

        if hard_edge:
             _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
             kernel = np.ones((3,3), np.uint8)
             warped_mask = cv2.dilate(warped_mask, kernel, iterations=1)

        if len(base_img.shape) == 3 and len(warped_img.shape) == 3:
            inv_mask = cv2.bitwise_not(warped_mask)
            
            img1_bg = cv2.bitwise_and(base_img, base_img, mask=inv_mask)
            
            img2_fg = cv2.bitwise_and(warped_img, warped_img, mask=warped_mask)
            
            result_img = cv2.add(img1_bg, img2_fg)
            return result_img
        
        return base_img

    def paste_image_with_alpha(self, base_img, overlay_img, target_points, antialiasing=False):
        if len(target_points) != 4:
            raise ValueError('need 4 point required')

        dst_pts = np.array(target_points, dtype=np.float32)
        h, w = overlay_img.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        bh, bw = base_img.shape[:2]
        warped_overlay = cv2.warpPerspective(overlay_img, M, (bw, bh), 
                                         flags=cv2.INTER_LINEAR if antialiasing else cv2.INTER_NEAREST)

        if warped_overlay.shape[2] == 4:
            b_overlay, g_overlay, r_overlay, a_overlay = cv2.split(warped_overlay)
            overlay_rgb = cv2.merge((b_overlay, g_overlay, r_overlay))
            mask = a_overlay
        else:
            overlay_rgb = warped_overlay
            mask_src = np.zeros((h, w), dtype=np.uint8)
            mask_src[:] = 255
            mask = cv2.warpPerspective(mask_src, M, (bw, bh), 
                                          flags=cv2.INTER_LINEAR if antialiasing else cv2.INTER_NEAREST)

        if base_img.shape[2] == 3:
            base_float = base_img.astype(float)
            overlay_float = overlay_rgb.astype(float)
            alpha_float = mask.astype(float) / 255.0
            
            alpha_float = cv2.merge([alpha_float, alpha_float, alpha_float])
            
            result_float = (overlay_float * alpha_float) + (base_float * (1.0 - alpha_float))
            return result_float.astype(np.uint8)
        
        elif base_img.shape[2] == 4:
            pass

        return base_img

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        ui_file = QFile('main.ui')
        if not ui_file.open(QIODevice.ReadOnly):
            print('can not open ui file')
        self.window = loader.load(ui_file)
        ui_file.close()
        self.window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.window.setAttribute(Qt.WA_TranslucentBackground)
        self.window.setWindowTitle('Skybox To Bedrock')
        self.path = None
        self.move_y = 0
        self.background = self.window.findChild(QLabel, 'background')
        self.shadowBG = QGraphicsDropShadowEffect(self)
        self.shadowBG.setOffset(0, 2)
        self.shadowBG.setBlurRadius(30)
        self.shadowBG.setColor(QColor(0, 0, 0, 255))
        self.background.setGraphicsEffect(self.shadowBG)
        self.title_bar = self.window.findChild(QFrame, 'move')
        if self.title_bar:
            self.title_bar.installEventFilter(self)
        self.close_btn = self.window.findChild(QPushButton, 'closeButton')
        self.close_btn.clicked.connect(self.window.close)
        self.min_btn = self.window.findChild(QPushButton, 'minimizeButton')
        self.min_btn.clicked.connect(self.window.showMinimized)
        self.picker = self.window.findChild(QPushButton, 'picker')
        self.picker.clicked.connect(self.pick_image)
        self.show_image = self.window.findChild(QLabel, 'show_image')
        self.option = self.window.findChild(QListWidget, 'listWidget')
        self.option.currentTextChanged.connect(self.option_changed)
        self.exp_btn = self.window.findChild(QPushButton, 'export_button')
        self.exp_btn.clicked.connect(self.exp)

    def exp(self):
        if self.path is None or not os.path.exists(self.path):
            print("no file selected or file does not exist")
            QMessageBox.critical(self.window, "Error", "No file selected or file does not exist")
            return

        target_move_y = self.move_y
        if target_move_y == 999:
            target_move_y = 0

        try:
            fixer = SkyboxFixer()
            fixer.skybox_fix_main(self.path, target_move_y)
            
            exp_path, selected_filter = QFileDialog.getSaveFileName(self.window, '导出文件', './转换结果.zip', 'zip文件 (*.zip)')
            if exp_path:
                shutil.copyfile('转换结果.zip', exp_path)
                QMessageBox.information(self.window, "成功", "导出成功！")
        except Exception as e:
            print(f"导出发生错误: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.window, "错误", f"导出失败: {str(e)}")
        return

    def option_changed(self, i: str):
        if i == '国际基岩版':
            self.move_y = 0.206
        elif i == '网易基岩版':
            self.move_y = 0.133
        return None

    def eventFilter(self, obj, event):
        if obj == self.title_bar and event.type() == QEvent.MouseButtonPress and (event.button() == Qt.LeftButton):
            window_handle = self.window.windowHandle()
            window_handle.startSystemMove()
        return super().eventFilter(obj, event)

    def pick_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.window, '选择文件', '', '图片 (*.png;*.jpg;*.bmp)')
        if file_path:
            self.show_image.setStyleSheet(f"image:url('{file_path}');padding:3px;")
            self.path = file_path
        return None

class SkyboxFixer:

    def skybox_fix_main(self, path, move_y):
        self.min_size_of_one = 128
        self.mix = PerspectiveImagePaster.paste_image_with_alpha
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        height, width, channels = image.shape
        min_size = self.min_size_of_one
        if width % 2 == 0 and height % 2 == 0 and (3 * height == 2 * width):
            if height >= 2 * min_size:
                image_picked = image
                size = height / 2
        size = int(size)
        self.tImg = np.full((size, size, 3), (0, 0, 0), dtype=np.uint8)
        t = self.tImg
        pieces_base = []
        for cy in range(2):
            for cx in range(3):
                pieces_base.append(image_picked[size * cy:size * (cy + 1), size * cx:size * (cx + 1)])
        
        pieces = [pieces_base[4], pieces_base[5], pieces_base[2], pieces_base[3], pieces_base[1], pieces_base[0]]
        
        if move_y == 0:
            result = pieces
        else:
            my_px = size * move_y
            
            z_value = my_px
            
            result = [None] * 6
            
            result[0] = self.draw_side(size, pieces[4], pieces[0], pieces[5], my_px)

            result[1] = self.draw_side(size, cv2.rotate(pieces[4], cv2.ROTATE_90_CLOCKWISE), pieces[1], pieces[5], my_px)

            result[2] = self.draw_side(size, cv2.rotate(pieces[4], cv2.ROTATE_180), pieces[2], pieces[5], my_px)

            result[3] = self.draw_side(size, cv2.rotate(pieces[4], cv2.ROTATE_90_COUNTERCLOCKWISE), pieces[3], pieces[5], my_px)
            
            result[4] = self.process_top(size, 
                pieces[4],
                pieces[0],
                pieces[2],
                pieces[3],
                pieces[1],
                z_value)
            
            z_int = int(math.ceil(z_value))
            if z_int < 1: z_int = 1
            if z_int > size: z_int = size
            
            strip_back = pieces[0][size - z_int : size, :]
            strip_front = pieces[2][size - z_int : size, :]
            strip_left = pieces[3][size - z_int : size, :]
            strip_right = pieces[1][size - z_int : size, :]
            
            result[5] = self.process_bottom(size, 
                pieces[5],
                strip_back,
                strip_front,
                strip_left,
                strip_right,
                my_px)

        for i in range(6):
            cv2.imwrite(f'cubemap_{i}.png', result[i])
        with zipfile.ZipFile('转换结果.zip', 'w') as zipobj:
            for i in range(6):
                zipobj.write(f'cubemap_{i}.png')
    pass

    def process_perspective_tile(self, size, image_center, neighbors, my_px):
        offset = float(my_px)
        if offset >= size / 2:
            offset = size / 2 - 1.0
        
        temp_img = np.full((size, size, 3), (0, 0, 0), dtype=np.uint8)
        paster = PerspectiveImagePaster()
        
        pts_center = np.float32([
            [offset, offset],
            [size - offset, offset],
            [offset, size - offset],
            [size - offset, size - offset]
        ])
        temp_img = paster.paste_image_by_perspective(temp_img, image_center, pts_center, antialiasing=True, hard_edge=True)
        
        if neighbors[0] is not None:
            pts_top = np.float32([
                [0, 0], 
                [size, 0], 
                [offset, offset], 
                [size - offset, offset]
            ])
            temp_img = paster.paste_image_by_perspective(temp_img, neighbors[0], pts_top, antialiasing=True, hard_edge=True)

        if neighbors[1] is not None:
            pts_bottom = np.float32([
                [offset, size - offset], 
                [size - offset, size - offset], 
                [0, size], 
                [size, size]
            ])
            temp_img = paster.paste_image_by_perspective(temp_img, neighbors[1], pts_bottom, antialiasing=True, hard_edge=True)

        if neighbors[2] is not None:
            pts_left = np.float32([
                [0, 0], 
                [offset, offset], 
                [0, size], 
                [offset, size - offset]
            ])
            temp_img = paster.paste_image_by_perspective(temp_img, neighbors[2], pts_left, antialiasing=True, hard_edge=True)

        if neighbors[3] is not None:
            pts_right = np.float32([
                [size - offset, offset], 
                [size, 0], 
                [size - offset, size - offset], 
                [size, size]
            ])
            temp_img = paster.paste_image_by_perspective(temp_img, neighbors[3], pts_right, antialiasing=True, hard_edge=True)

        return temp_img

    def process_top(self, size, image_m, img_back, img_front, img_left, img_right, my_px):
        offset = int(my_px)
        if offset <= 0:
            return image_m
            
        if 2 * offset >= size:
             return image_m
             
        cropped = image_m[offset : size - offset, offset : size - offset]
        result = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
        return result

    def process_bottom(self, size, image_m, img_back, img_front, img_left, img_right, my_px):
        n_top = img_back
        n_bottom = cv2.rotate(img_front, cv2.ROTATE_180)
        n_left = cv2.rotate(img_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
        n_right = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)
        
        return self.process_perspective_tile(size, image_m, [n_top, n_bottom, n_left, n_right], my_px)

    def draw_side(self, size, image_t, image_m, image_b, my_px):
        offset = float(my_px)
        offset_int = int(math.ceil(offset))
        
        temp_img = np.full((size, size, 3), (0, 0, 0), dtype=np.uint8)
        paster = PerspectiveImagePaster()
        
        pts_center = np.float32([
            [0, offset],
            [size, offset],
            [0, size + offset],
            [size, size + offset]
        ])
        temp_img = paster.paste_image_by_perspective(temp_img, image_m, pts_center, antialiasing=True, hard_edge=True)
        
        if offset > 0 and image_t is not None:
            z = offset_int
            if z < 1: z = 1
            if z >= size: z = size - 1
            
            top_strip = image_t[size - z : size, :]
            
            h_strip, w_strip = top_strip.shape[:2]
            src_pts = [
                [offset, 0],
                [size - offset, 0],
                [0, h_strip],
                [size, h_strip]
            ]
            
            pts_dst_top = np.float32([
                [0, 0],
                [size, 0],
                [0, offset],
                [size, offset]
            ])
            
            temp_img = paster.paste_image_by_perspective(temp_img, top_strip, pts_dst_top, antialiasing=True, hard_edge=True, src_points=src_pts)
            
        return temp_img

if __name__ == '__main__':
    loader = QUiLoader()
    app = QApplication([])
    app.setWindowIcon(QtGui.QIcon('_internal/ico.ico'))
    main_window = MainWindow()
    main_window.window.show()
    app.exec()
