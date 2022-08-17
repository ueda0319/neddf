# Neural Density-Distance Fields

### [Project Page](https://ueda0319.github.io/neddf/) | [arXiv](https://arxiv.org/abs/2207.14455)

PyTorch implementation of Neural Distance-Density Field (NeDDF), a 3D representation that reciprocally constrains the distance and density fields.

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

This repository is based on [Ascender project](https://github.com/cvpaperchallenge/Ascender).
Please refer to Ascender for detailed instructions on how to set up the host environment.

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
$ poetry run python neddf/scripts/run_eval.py {pretrained files directory}
```

For example, use following commands to evaluate neddf model in bunny_smoke scene.
```bash
$ poetry run python neddf/scripts/run_eval.py pretrained/bunny_smoke/
```

You can download pretrained models of nerf_synthetic_dataset from [this link](https://drive.google.com/file/d/1YJnky8bye0WU-_yZbC0DCiF0rm-36s57/view?usp=sharing)

## Visualize pretrained fields
To visualize trained distance, density, aux.gradient and color fields, please run following command.
```bash
$ poetry run python neddf/scripts/fields_visualizer.py {pretrained files directory}
```
This visualizer draw fields by 2D slices. Following video is visualization example in lego scene(`pretrained/lego`)


https://user-images.githubusercontent.com/26667016/184655073-1ca5f55c-0170-4c61-96aa-802aa9ccfc88.mp4



## Visualize for check dataset
To visualize dataset, please run following command (override of dataset config is optional):
```bash
$ poetry run python neddf/scripts/dataset_visualizer.py dataset.dataset_dir=data/bunny_smoke/
```
