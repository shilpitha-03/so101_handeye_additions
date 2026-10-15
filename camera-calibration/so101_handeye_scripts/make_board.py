"""Generate a printable ChArUco board PNG from config.py (300 DPI). Print at 100% / Actual size."""
import cv2 as cv
from PIL import Image
import config
from common import create_charuco_board

DPI = 300
px_per_square = round(config.SQUARE_LENGTH_M / 0.0254 * DPI)
margin = round(0.010 / 0.0254 * DPI)  # 10 mm white border
size = (config.SQUARES_X * px_per_square + 2 * margin, config.SQUARES_Y * px_per_square + 2 * margin)
img = create_charuco_board().generateImage(size, marginSize=margin)
Image.fromarray(img).save("charuco_board.png", dpi=(DPI, DPI))
print(f"Saved charuco_board.png: {config.SQUARES_X}x{config.SQUARES_Y} squares, "
      f"{config.SQUARE_LENGTH_M*1000:.1f} mm squares, {config.MARKER_LENGTH_M*1000:.1f} mm markers, {config.ARUCO_DICT_NAME}")
print("Print at 100% / 'Actual size'. Then MEASURE a square with calipers and update config.py.")
