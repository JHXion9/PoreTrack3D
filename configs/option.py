'''
Description: 
Author: jiahaoxiong
Version: 
Date: 2024-11-20 09:23:15
LastEditors: i_xiongjiahao@cvte.com
LastEditTime: 2024-11-26 01:56:59
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-id", "--people_id", type=str, default = '033' , help="人员id")
    # parser.add_argument("--seq_name", type=str, default = 'EMO-3-angry+sad', help="序列名")

    # parser.add_argument("--first_xy", nargs = 2,type=float, default = [1569,1614], help="初始位置跟踪位置")
    # parser.add_argument("--details_pos", type=str, default = 'pore6', help="轨迹生成部位具体位置  left_face, right_face, top_face")

    # parser.add_argument("--base_view", type=int, default = 11, help="正脸视角对应的cam_id")

    # parser.add_argument("--frame_num", type=int, default =150, help="跟踪总帧数")
    # parser.add_argument("--patch_radius", type=int, default = 100, help="patch半径")

    # parser.add_argument("--position", type=str, default = 'pore', help="轨迹生成部位  eyes, mole, pore, mouse")

    parser.add_argument("-id", "--people_id", type=str, default = '124' , help="人员id")
    parser.add_argument("--seq_name", type=str, default = 'EMO-2-surprise+fear', help="序列名")

    parser.add_argument("--first_xy", nargs = 2,type=float, default = [755,1808], help="初始位置跟踪位置")
    parser.add_argument("--details_pos", type=str, default = 'pore13', help="轨迹生成部位具体位置  left_face, right_face, top_face")

    parser.add_argument("--base_view", type=int, default = 9, help="正脸视角对应的cam_id")

    parser.add_argument("--frame_num", type=int, default =150, help="跟踪总帧数")
    parser.add_argument("--patch_radius", type=int, default = 100, help="patch半径")

    parser.add_argument("--position", type=str, default = 'pore', help="轨迹生成部位  eyes, mole, pore, mouse")


    



    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt