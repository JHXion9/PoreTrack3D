# PoreTrack3D: A Benchmark for Dynamic 3D Gaussian Splatting in Pore-Scale Facial Trajectory Tracking
---

## Project Overview  
PoreTrack3D is the first benchmark designed for dynamic 3D Gaussian Splatting in pore-scale, non-rigid facial trajectory tracking.
The dataset comprises over 440,000 facial trajectories, including 52,000 sequences longer than 10 frames and 68 carefully reviewed trajectories covering the full 150-frame sequence.

Unlike traditional benchmarks that focus only on coarse facial landmarks, PoreTrack3D captures both macro facial landmarks and pore-scale keypoint trajectories, enabling detailed analysis of subtle skin-surface motion and fine-grained facial expressions.

We conduct a systematic evaluation of state-of-the-art dynamic 3D Gaussian Splatting methods, providing the first performance baseline for this emerging domain.
The dataset creation pipeline also establishes a new framework for high-fidelity facial motion capture and dynamic 3D reconstruction, paving the way for future research in micro-expression analysis and realistic face modeling.

---


## Trajectory Visualization
<!-- <p align="center">
  <img src="assets/063.gif" width="40%">
  <img src="assets/056.gif" width="40%">
</p> -->


https://github.com/user-attachments/assets/cb396a0e-c05b-4f53-b6a3-f29c9037bc74


You can run `python ./scripts/review.py` to perform a manual review.




---

## Dataset  

The full dataset can be downloaded from the [project page](https://drive.google.com/drive/folders/1M3v9vuxgaG287RORreggHuHo9efBLHLe?usp=sharing), under [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset includes:

```
├── data
│   | 031 
│     ├── initial_pcd.ply            # The initial 3D point cloud                
│     ├── _xyz.json                  # 150-frame 3D trajectory for all positions.
│     | mesh                         # Per-frame reconstructed mesh files
│     | left_eye
│       | manualreview               # for review.py
│       ├── _xyz.json                # The 150-frame 3D trajectory of this position.
│       ├── all_trajectory2D.json    # All 2D trajectories of this position.
│       ├── all_trajectory3D.json    # All 3D trajectories of this position.
│     | pore
│       | manualreview               # for review.py
│       ├── _xyz.json                # The 150-frame 3D trajectory of this position.
│       ├── all_trajectory2D.json    # All 2D trajectories of this position.
│       ├── all_trajectory3D.json    # All 3D trajectories of this position.
│     |  ...
│   | 033 
│     ├── initial_pcd.ply            # The initial 3D point cloud 
│     ├── _xyz.json                  # 150-frame 3D trajectory for all positions.
│     | mesh                         # Per-frame reconstructed mesh files
│     | left_eye
│       | manualreview               # for review.py
│       ├── _xyz.json                # The 150-frame 3D trajectory of this position.
│       ├── all_trajectory2D.json    # All 2D trajectories of this position.
│       ├── all_trajectory3D.json    # All 3D trajectories of this position.
│     | pore
│       | manualreview               # for review.py
│       ├── _xyz.json                # The 150-frame 3D trajectory of this position.
│       ├── all_trajectory2D.json    # All 2D trajectories of this position.
│       ├── all_trajectory3D.json    # All 3D trajectories of this position.
│     |  ...
│   | ...

```

Additional image data and camera parameters can be requested through the [NeRSemble](https://tobias-kirschstein.github.io/nersemble/)




---
