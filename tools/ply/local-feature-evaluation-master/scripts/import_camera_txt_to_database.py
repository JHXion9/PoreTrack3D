# This script is based on an original implementation by True Price.
# Created by liminghao
import sys
import numpy as np
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--database_path", default="database.db")
    # args = parser.parse_args()
    # if os.path.exists(args.database_path)==False:
    #     print("ERROR: database path dosen't exist -- please check database.db.")
    #     return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()


def print_camera_table(database_path):
    """
    Prints the contents of the 'cameras' table in a COLMAP database.

    Args:
        database_path (str): Path to the COLMAP database (database.db).
    """

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM cameras")
        rows = cursor.fetchall()

        if not rows:
            print("Error: No cameras found in the database.")
            return

        print("Contents of the 'cameras' table:")
        print("--------------------------------------------------")
        print("camera_id | model | width | height | params | prior_focal_length")
        print("--------------------------------------------------")

        for row in rows:
            camera_id, model, width, height, params_blob, prior_focal_length = row
            params = np.frombuffer(params_blob, dtype=np.float64)
            print(f"{camera_id:9} | {model:5} | {width:5} | {height:6} | {params} | {prior_focal_length}") # Adjusted formatting

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def print_camera_poses(database_path):
    """
    Prints the image names and camera poses (rotation and translation) from the COLMAP database.
    Handles None values in pose data.

    Args:
        database_path (str): Path to the COLMAP database (database.db).
    """

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images")
        rows = cursor.fetchall()

        if not rows:
            print("Error: No images found in the database.")
            return

        print("Camera Poses from the 'images' table:")
        print("-----------------------------------------------------------------------------------------------------------------")
        print("image_id | name                          | camera_id |   qw   |   qx   |   qy   |   qz   |   tx   |   ty   |   tz   ")
        print("-----------------------------------------------------------------------------------------------------------------")

        for row in rows:
            image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz = row

            # Handle None values
            qw = qw if qw is not None else 0.0
            qx = qx if qx is not None else 0.0
            qy = qy if qy is not None else 0.0
            qz = qz if qz is not None else 0.0
            tx = tx if tx is not None else 0.0
            ty = ty if ty is not None else 0.0
            tz = tz if tz is not None else 0.0

            print(f"{image_id:8} | {name:30} | {camera_id:9} | {qw:6.3f} | {qx:6.3f} | {qy:6.3f} | {qz:6.3f} | {tx:6.3f} | {ty:6.3f} | {tz:6.3f}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

import pycolmap

def print_camera_info(cameras_path):
    """
    读取并打印 cameras.bin 文件的内容。

    Args:
        cameras_path: cameras.bin 文件的路径。
    """
    try:
        cameras = pycolmap.read_cameras_binary(cameras_path)
        for camera_id, camera in cameras.items():
            print(f"Camera ID: {camera_id}")
            print(f"  Model: {camera.model_name()}")  # 输出模型名称，如 'SIMPLE_PINHOLE'
            print(f"  Width: {camera.width}")
            print(f"  Height: {camera.height}")
            print(f"  Params: {camera.params}")
            print(f"  Params Info: {camera.params_info()}") # 输出参数的意义
            print("-" * 20)
    except FileNotFoundError:
        print(f"错误：找不到文件 {cameras_path}")
    except Exception as e:
        print(f"发生错误: {e}")



if __name__ == "__main__":
    camTodatabase("/media/DGST_data/Test_Data/030-/EMO-1-shout+laugh/1/psiftproject/sparse/0/cameras.txt", "/media/DGST_data/Test_Data/030/EMO-1-shout+laugh/1/psiftproject/database.db")
    # print_camera_table("/media/DGST_data/Test_Data/030/EMO-1-shout+laugh/1/psiftproject/database.db")
    # print_camera_poses("/media/DGST_data/Test_Data/030/EMO-1-shout+laugh/1/psiftproject/database.db")
    # print_camera_info("/media/DGST_data/Test_Data/030/EMO-1-shout+laugh/1/psiftproject/sparse/cameras.bin")