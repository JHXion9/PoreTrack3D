# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

from __future__ import print_function, division

import os
import sys
import glob
import argparse
import sqlite3
import subprocess
import multiprocessing
import json
import numpy as np
from import_camera_txt_to_database import camTodatabase

IS_PYTHON3 = sys.version_info[0] >= 3

#python reconstruction_pipeline.py --dataset_path /media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/ETH/datasets/datasets/Fountain --colmap_path /media/amax/ef914467-feed-4743-b2ec-58c3abb24e7a/LHW/ETH/colmap/build/src/exe
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='/media/human_face_need_test/brown_v2/018/EMO-1-shout+laugh/10/psiftproject',
                        )
    parser.add_argument("--colmap_path", default='/media/DenseGSTracking/colmap/build/src/colmap/exe'
                             )
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def import_matches(args):
    # 连接数据库，删除对应的keypoints，descriptors，matches
    print(args.dataset_path)
    connection = sqlite3.connect(os.path.join(args.dataset_path, "database.db"))
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master "
                   "WHERE type='table' AND name='inlier_matches';")
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    if inlier_matches_table_exists:
        cursor.execute("DELETE FROM inlier_matches;")
    else:
        cursor.execute("DELETE FROM two_view_geometries;")
    connection.commit()

    images = {}
    # 获取所有图片的名字
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    #  导入keypoints
    for image_name, image_id in images.items():
        
        print("Importing features for", image_name)
        keypoint_path = os.path.join(args.dataset_path, "keypoints",
                                     image_name + ".bin")
        keypoints = read_matrix(keypoint_path, np.float32)
        descriptor_path = os.path.join(args.dataset_path, "descriptors",
                                     image_name + ".bin")
        descriptors = read_matrix(descriptor_path, np.float32)
        assert keypoints.shape[1] == 4
        #assert keypoints.shape[0] == descriptors.shape[0]
        if IS_PYTHON3:
            keypoints_str = keypoints.tobytes()
        else:
            keypoints_str = np.getbuffer(keypoints)
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints_str))
        connection.commit()

    # 遍历所有的match 文件
    image_pairs = []
    image_pair_ids = set()
    for match_path in glob.glob(os.path.join(args.dataset_path,
                                             "matches/*---*.bin")):
        image_name1, image_name2 = \
            os.path.basename(match_path[:-4]).split("---")
        image_pairs.append((image_name1, image_name2))
        print("Importing matches for", image_name1, "---", image_name2)
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        #遍历执行每个match 数据导入文件中
        image_pair_ids.add(image_pair_id)
        matches = read_matrix(match_path, np.uint32)
        assert matches.shape[1] == 2
        if IS_PYTHON3:
            matches_str = matches.tobytes()
        else:
            matches_str = np.getbuffer(matches)
        cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1],
                        matches_str))
        connection.commit()
    # 写文件image_pair  导入了哪个对
    with open(os.path.join(args.dataset_path, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    cursor.close()
    connection.close()


    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "matches_importer",
                     "--database_path",
                     os.path.join(args.dataset_path, "database.db"),
                     "--SiftMatching.num_threads", "3",  # debug
                     "--match_list_path",
                     os.path.join(args.dataset_path, "image-pairs.txt"),
                     "--match_type", "pairs"])

    connection = sqlite3.connect(os.path.join(args.dataset_path, "database.db"))
    cursor = connection.cursor()

    cursor.execute("SELECT count(*) FROM images;")
    num_images = next(cursor)[0]

    cursor.execute("SELECT count(*) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_pairs = next(cursor)[0]

    cursor.execute("SELECT sum(rows) FROM two_view_geometries WHERE rows > 0;")
    num_inlier_matches = next(cursor)[0]

    cursor.close()
    connection.close()

    return dict(num_images=num_images,
                num_inlier_pairs=num_inlier_pairs,
                num_inlier_matches=num_inlier_matches)


def reconstruct(args):
    database_path = os.path.join(args.dataset_path, "database.db")
    image_path = os.path.join(args.dataset_path, "images")
    sparse_path = os.path.join(args.dataset_path, "sparse")
    dense_path = os.path.join(args.dataset_path, "dense")
    created_sparse_path = os.path.join(args.dataset_path, "created/sparse")

    if not os.path.exists(sparse_path):
        os.makedirs(os.path.join(sparse_path,'0'))
        os.makedirs(os.path.join(sparse_path,'1'))
    if not os.path.exists(dense_path):
        os.makedirs(dense_path)

    if args.frame == 1:
        print("开始导入相机参数")
        camTodatabase(os.path.join(created_sparse_path, "cameras.txt"), database_path)
        print("导入相机参数完成")
        with open('./log_file_point_triangulator.txt', "w") as f_pt:
            subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "point_triangulator",
                            "--database_path", database_path,
                            "--image_path", image_path,
                            "--input_path", created_sparse_path,
                            "--output_path", os.path.join(sparse_path,'0'),
                            ],
                            stdout=f_pt, stderr=f_pt)
        
        with open('./log_file_bundle_adjuster.txt', "w") as f_ba:
            subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "bundle_adjuster",
                            "--input_path", os.path.join(sparse_path,'0'),
                            "--output_path", os.path.join(sparse_path,'1'),
                            "--BundleAdjustment.refine_focal_length", '1',
                            "--BundleAdjustment.refine_principal_point", '1',
                            "--BundleAdjustment.refine_extrinsics", '1',
                            "--BundleAdjustment.refine_extra_params", '0',
                            ],
                            stdout=f_ba, stderr=f_ba) # 重定向 stdout 和 stderr 到文件

        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "model_converter",
                        "--input_path", os.path.join(sparse_path,'1'),
                        "--output_path", os.path.join(sparse_path,'1'),
                        "--output_type", "TXT"])
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "model_converter",
                            "--input_path", os.path.join(sparse_path,'1'),
                            "--output_path", os.path.join(sparse_path, "sparse.ply"),
                            "--output_type", "PLY"])
        
    else:
        print("开始导入相机参数")
        camTodatabase(os.path.join(created_sparse_path, "cameras.txt"), database_path)
        print("导入相机参数完成")

        with open('./log_file_point_triangulator.txt', "w") as f_pt:
            subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "point_triangulator",
                            "--database_path", database_path,
                            "--image_path", image_path,
                            "--input_path", created_sparse_path,
                            "--output_path", os.path.join(sparse_path,'0'),
                            ],
                            stdout=f_pt, stderr=f_pt)
        
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "bundle_adjuster",
                            "--input_path", os.path.join(sparse_path,'0'),
                            "--output_path", os.path.join(sparse_path,'1'),
                            "--BundleAdjustment.refine_focal_length", '0',
                            "--BundleAdjustment.refine_principal_point", '0',
                            "--BundleAdjustment.refine_extrinsics", '0',
                            "--BundleAdjustment.refine_extra_params", '0',
                            ]
                            ) # 重定向 stdout 和 stderr 到文件
        
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "model_converter",
                        "--input_path", os.path.join(sparse_path,'1'),
                        "--output_path", os.path.join(sparse_path,'1'),
                        "--output_type", "TXT"])
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                            "model_converter",
                            "--input_path", os.path.join(sparse_path,'1'),
                            "--output_path", os.path.join(sparse_path, "sparse.ply"),
                            "--output_type", "PLY"])
        
    largest_model_path = os.path.join(sparse_path,'1')
    print("Largest model path:", largest_model_path)
    workspace_path = os.path.join(dense_path,'0')
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    # Run the dense reconstruction.
    with open('./log_file_image_undistorter.txt', "w") as f_ut:
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "image_undistorter",
                        "--image_path", image_path,
                        "--input_path", largest_model_path,
                        "--output_path", workspace_path,
                        "--max_image_size", "1200",
                        ],
                            stdout=f_ut, stderr=f_ut)
        
    with open('./log_file_patch_match_stereo.txt', "w") as f_ut:
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "patch_match_stereo",
                        "--workspace_path", workspace_path,
                        "--PatchMatchStereo.geom_consistency", "false",
                       ],
                            stdout=f_ut, stderr=f_ut)
        
    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "model_converter",
                        "--input_path", os.path.join(workspace_path,'sparse'),
                        "--output_path", os.path.join(workspace_path,'sparse'),
                        "--output_type", "TXT"])

    with open('./log_file_stereo_fusion.txt', "w") as f_ut:
        subprocess.call([os.path.join(args.colmap_path, "colmap"),
                        "stereo_fusion",
                        "--workspace_path", workspace_path,
                        "--input_type", "photometric",
                        "--output_path", os.path.join(workspace_path, "fused.ply"),
                        ],
                            stdout=f_ut, stderr=f_ut)
    
    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                    "model_converter",
                    "--input_path", os.path.join(dense_path,'sparse'),
                    "--output_path", os.path.join(dense_path,'sparse'),
                    "--output_type", "TXT"])

    stats = subprocess.check_output(
        [os.path.join(args.colmap_path, "colmap"), "model_analyzer",
         "--path", largest_model_path])

    stats = stats.decode().split("\n")

    for stat in stats:
        if stat.startswith("Registered images"):
            num_reg_images = int(stat.split()[-1])
        elif stat.startswith("Points"):
            num_sparse_points = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            num_observations = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            mean_track_length = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            num_observations_per_image = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            mean_reproj_error = float(stat.split()[-1][:-2])

    with open(os.path.join(workspace_path, "fused.ply"), "rb") as fid:
        line = fid.readline().decode()
        while line:
            if line.startswith("element vertex"):
                num_dense_points = int(line.split()[-1])
                break
            line = fid.readline().decode()

    return dict(
                num_dense_points=num_dense_points)


def main():
    args = parse_args()

    matching_stats = import_matches(args)
    reconstruction_stats = reconstruct(args)
    
    print(json.dumps(matching_stats))
    # print()
    # print(78 * "=")
    # print("Raw statistics")
    # print(78 * "=")
    # print(matching_stats)
    # print(reconstruction_stats)

    # print()
    # print(78 * "=")
    # print("Formatted statistics")
    # print(78 * "=")
    # print("| " + " | ".join(
    #         map(str, [os.path.basename(args.dataset_path),
    #                   "METHOD",
    #                   matching_stats["num_images"],
                      
    #                   reconstruction_stats["num_dense_points"],
    #                   "",
    #                   "",
    #                   "",
    #                   "",
    #                   matching_stats["num_inlier_pairs"],
    #                   matching_stats["num_inlier_matches"]])) + " |")


if __name__ == "__main__":
    main()
