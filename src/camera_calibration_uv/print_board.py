import os
import argparse
import cv2
import matplotlib.pyplot as plt

from .charuco_utils import create_board, draw_board

PAPER_SIZES_INCH = {
    'A4':  (8.27, 11.69),
    'A3':  (11.69, 16.54),
    'A2':  (16.54, 23.39),
    'A1':  (23.39, 33.11),
    'A0':  (33.11, 46.81)
}

def calc_fit_size_px(board: cv2.aruco.CharucoBoard, dpi: int = 300) -> tuple: 
    squares_x, squares_y = board.getChessboardSize()
    square_length_m = board.getSquareLength()

    inch_per_meter = 39.3701
    width_inch = squares_x * square_length_m * inch_per_meter
    height_inch = squares_y * square_length_m * inch_per_meter

    size_inch = None
    for name, (pw, ph) in PAPER_SIZES_INCH.items():
        if pw >= width_inch and ph >= height_inch:
            size_inch = (pw, ph)
            print(f'Using {name} {size_inch} inches')
            break
    if size_inch is None:
        raise ValueError(f"No paper format fits")
    
    width_px, height_px = (int(dim * dpi) for dim in size_inch)
    return (width_px, height_px), size_inch

def print_board(args): 
    board = create_board(args.board)
    size_px, size_inch = calc_fit_size_px(board, dpi=args.dpi)
    img = draw_board(
        board=board,
        size_px=size_px,
        margin_px=args.margin_px,
        border_bits=args.border_bits
    )

    base_path, _ = os.path.splitext(args.board)
    board_file_path = base_path + '.png'
    cv2.imwrite(board_file_path, img)
    print(f"Board saved at {board_file_path}, size: {img.shape[:2]} pixels")

def main(): 
    parser = argparse.ArgumentParser(description="Print charuco board to image file")
    parser.add_argument('--board', type=str, default='example_boards/charuco_5x7_A4.yaml', help='YAML File describing charuco board')
    parser.add_argument('--margin-px', type=int, default=0, help='Minimum margins (in pixels) of the board in the output image')
    parser.add_argument('--border-bits', type=int, default=1, help='Width of the marker borders')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for the output image')
    args = parser.parse_args()

    print_board(args)
