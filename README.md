# SceneForge: GPU-Accelerated 3D Reconstruction from Monocular Video

SceneForge reconstructs continuous monocular video into a shared 3D map by combining
learned keyframe selection, VGGT-Ω geometry priors, and GPU-accelerated bundle
adjustment. The system aligns overlapping reconstructions, jointly refines camera
poses, calibration, and landmarks using temporal and verified loop correspondences,
and fuses refined depth into a colored dense point cloud. Live Rerun visualization
reveals the reconstruction as it develops, from selected keyframes to the final map.

## 🖥️ Tested Configuration

SceneForge has been tested on:

- 🐧 **Ubuntu:** 24.04
- 🧠 **GPUs:** 2 × NVIDIA RTX A6000, with 48 GB VRAM per GPU
- ⚙️ **CUDA (host):** 13.3
- 🧊 **Environment:** Docker with NVIDIA Container Toolkit and GPU support

> This is the tested reference configuration. Memory requirements depend on image
> resolution and the number of keyframes per VGGT-Ω window.

## 🚀 Quick Start

Install Git, Docker with NVIDIA Container Toolkit, and FFmpeg (`ffprobe`) on the
host before starting. Run these commands from the host terminal.

1. **Clone the repository and its submodules**

   ```bash
   git clone --recurse-submodules https://github.com/jagennath-hari/SceneForge.git
   cd SceneForge
   ```

2. **Set up Hugging Face access**

   Request access to [VGGT-Ω](https://huggingface.co/facebook/VGGT-Omega) and wait
   for approval. Create the token file:

   ```bash
   mkdir -p .secrets
   chmod 700 .secrets
   (umask 077; touch .secrets/hf_token)
   chmod 600 .secrets/hf_token
   nano .secrets/hf_token
   ```

   Paste only a read token from the approved account with access to the VGGT-Ω
   repository, then save the file. It is ignored by Git and mounted read-only.
   Model weights download automatically when needed.

3. **Reconstruct a video**

   ```bash
   bash scripts/build_and_start.sh "/path/to/your/video.mp4"
   ```

   The script validates the video, builds the Docker environment, and starts
   reconstruction with live Rerun visualization. Results are saved under
   `data/output/`.

> Use a continuous video without cuts. Replace the example path with any local
> video file; it does not need to be inside the repository. Quote paths containing spaces.

## 📖 Citation

If you find SceneForge useful in your research, please consider citing this
software and the research papers that make the pipeline possible.

```bibtex
@misc{wang2026vggtomega,
  title         = {{VGGT-$\Omega$}},
  author        = {Jianyuan Wang and Minghao Chen and Shangzhan Zhang and Nikita Karaev and Johannes Schönberger and Patrick Labatut and Piotr Bojanowski and David Novotny and Andrea Vedaldi and Christian Rupprecht},
  year          = {2026},
  eprint        = {2605.15195},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.15195}
}
```

```bibtex
@article{selavprpp,
  title   = {{SelaVPR++}: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition},
  author  = {Lu, Feng and Jin, Tong and Lan, Xiangyuan and Zhang, Lijun and Liu, Yunpeng and Wang, Yaowei and Yuan, Chun},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  volume  = {48},
  number  = {3},
  pages   = {2731--2748},
  doi     = {10.1109/TPAMI.2025.3629287}
}
```

```bibtex
@inproceedings{shenoi2026raco,
  title     = {{RaCo}: Ranking and Covariance for Practical Learned Keypoints},
  author    = {Shenoi, Abhiram and Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle = {International Conference on 3D Vision},
  year      = {2026}
}
```

```bibtex
@article{Zhao2023ALIKED,
  title   = {{ALIKED}: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation},
  author  = {Zhao, Xiaoming and Wu, Xingming and Chen, Weihai and Chen, Peter C. Y. and Xu, Qingsong and Li, Zhengguo},
  journal = {IEEE Transactions on Instrumentation & Measurement},
  year    = {2023},
  volume  = {72},
  pages   = {1--16},
  doi     = {10.1109/TIM.2023.3271000}
}
```

```bibtex
@inproceedings{lindenberger2023lightglue,
  title     = {{LightGlue}: Local Feature Matching at Light Speed},
  author    = {Philipp Lindenberger and Paul-Edouard Sarlin and Marc Pollefeys},
  booktitle = {ICCV},
  year      = {2023}
}
```

## 🙏 Acknowledgement

This work integrates the following open-source libraries and tools:

- [**LightGlue-ONNX**](https://github.com/fabio-sim/LightGlue-ONNX) — ONNX/TensorRT implementation of the RaCo–ALIKED–LightGlue+ frontend.
- [**cuNLS**](https://github.com/nvidia-isaac/cuNLS) — GPU-accelerated nonlinear least-squares optimization for bundle adjustment.
