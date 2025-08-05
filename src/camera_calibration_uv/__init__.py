import argparse
from .intrinsic import calibrate_intrinsic

def calibrate() -> None:
    parser = argparse.ArgumentParser() 
    subparsers = parser.add_subparsers(title='commands', dest='command')

    parser_intrinsic = subparsers.add_parser('intrinsic', help='Calibrate intrinsic')
    parser_intrinsic.add_argument('--video-path', type=str, default='output/output.mp4', help='Path to video file')
    parser_intrinsic.add_argument('--board-path', type=str, default='example_boards/charuco_5x7.yaml', help='YAML File describing charuco board')
    parser_intrinsic.add_argument('--step-size', type=int, default=1, help='Use every n-th frame of video')
    parser_intrinsic.set_defaults(func=calibrate_intrinsic)

    args = parser.parse_args() 
    if args.command: 
        args.func(args)
    else: 
        parser.print_help() 

