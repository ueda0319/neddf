# Neural Density-Distance Fields

### [Project Page](https://ueda0319.github.io/neddf/) | [arXiv (coming soon)]()

PyTorch implementation of Neural Distance-Density Field (NeDDF), a 3D representation that reciprocally constrains the distance and density fields (CODE COMING SOON).

[Neural Density-Distance Fields](http://tancik.com/nerf)  
[Itsuki Ueda](https://sites.google.com/image.iit.tsukuba.ac.jp/itsukiueda)<sup>1</sup>,
[Yoshihiro Fukuhara](https://gatheluck.net)<sup>2</sup>,
[Hirokatsu Kataoka](http://hirokatsukataoka.net)<sup>3</sup>,
[Hiroaki Aizawa](https://aizawan.github.io)<sup>4</sup>,
[Hidehiko Shishido](https://sites.google.com/image.iit.tsukuba.ac.jp/shishido)<sup>1</sup>,
[Itaru Kitahara](https://sites.google.com/image.iit.tsukuba.ac.jp/kitahara)<sup>1</sup> <br>
<sup>1</sup>University of Tsukuba, <sup>2</sup>Waseda University, <sup>3</sup>National Institute of Advanced Industrial Science and Technology (AIST), <sup>4</sup>Hiroshima University
<br>
in ECCV 2022(poster)

## Prerequisite

- docker
- docker-compose
- nvidia-docker2

## How to setup docker container

```
## Move to the directory where docker-compose.yaml exists.
$ cd /path/to/neddf/environments/gpu

## build and start docker container
$ docker compose up -d core

## Enter docker container with bash
$ docker compose exec core bash
```

## How to run code
You can download the nerf_synthetic_dataset and nerf_llff_dataset proposed in original NeRF paper [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please place downloaded data to `data/`

### Train NeDDF

In the default configuration, the `bunny_smoke` dataset is set to learn.

Run
```bash
$ poetry run python neddf/scripts/run.py
```

To use other dataset with data format of `nerf_synthetic`, please select other dataset directory like follow:
```bash
$ poetry run python neddf/scripts/run.py dataset.dataset_dir="data/nerf_synthetic/drums/"
```

Logs are saved in `outputs/{datatime}/`.
Model parameters are saved to `outputs/{datatime}/models/`.
Rendering results are saved to `outputs/{datatime}/render/`, images rendered from a camera is in iteration number directory, and visualization of field slices are in `fields` directory.

### Evaluate NeDDF

Run
```bash
$ poetry run python neddf/scripts/run.py {pretrained files directory}
```

## Visualize for check dataset
To visualize dataset, please run following command (override of dataset config is optional):
```bash
$ poetry run python neddf/scripts/dataset_visualizer.py dataset=${DATASET_NAME}
```