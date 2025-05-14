ID=$1

python /media/Trajectory3D/tools/preprocess_for_trajectory.py -id $ID




# Run LLFF's imgs2poses.py
python /media/DenseGSTracking/LLFF/imgs2poses.py /media/colmap_temp/

# Copy LLFF's results
cp /media/colmap_temp/poses_bounds.npy /media/DGST_data/Data/$ID/poses_bounds_multipleview.npy

rm -rf /media/colmap_temp/