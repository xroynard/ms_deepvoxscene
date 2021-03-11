# [Classification of Point Cloud Scenes with Multiscale Voxel Deep Network](https://arxiv.org/abs/1804.03583)

Created by [Xavier Roynard](https://www.researchgate.net/profile/Xavier_Roynard) from [NPM3D team](http://caor-mines-paristech.fr/fr/recherche/3d-modeling/) of Centre for Robotics of Mines ParisTech. 

----------------------------------------------------------------

![Exemple de Segmentation et Classification](apercu_methode.png)

----------------------------------------------------------------

## Introduction
This work is based on our [arXiv paper](https://arxiv.org/abs/1804.03583), which were also presented in IROS 2018 workshop PPNIV'18 and can be found [here](https://project.inria.fr/ppniv18/files/2018/10/paper13.pdf). We proposed a multiscale convolutionnal network architecture for semantic segmentation of 3D point cloud scenes.

Point cloud is an important type of geometric data structure. Due to its irregular format, we transform such data to regular 3D voxel grids. This, however, renders data voluminous and avoids using high resolution grids, which forces to make a compromise between a low discretization step (allowing a better local representation of the surface) and a grid that represents a large volume (allowing a better understanding of the context).

In this paper, we use a multi-scale input to handle this problem. Our network, named MultiScale*N*\_DeepVoxScene (abbreviated in MS*N*\_DVS, *N* is the nuber of scales used), provides an architecture that allows a precise representation of the surface near a point and the more distant context around a point. Though simple, the use of several scales greatly improves classification results.

In this repository, we release code for training a multiscale classification 3D convolutionnal network on fully annotated point cloud scenes.

## Citation
If you find our work useful please cite one the following articles :

	@inproceedings{roynard2018classificationIROS,
	  title={Classification of Point Cloud for Road Scene Understanding with Multiscale Voxel Deep Network},
	  author={Roynard, Xavier and Deschaud, Jean-Emmanuel and Goulette, Fran{\c{c}}ois},
	  booktitle = {10th Workshop on Planning, Perception and Navigation for Intelligent Vehicles, PPNIV'18},
	  year={2018},
	  month={October},
	}
or

	@article{roynard2018classification,
	  title={Classification of Point Cloud Scenes with Multiscale Voxel Deep Network},
	  author={{Roynard}, {Xavier} and {Deschaud}, {Jean-Emmanuel} and {Goulette}, {François}},
	  journal={arXiv preprint arXiv:1804.03583},
	  year={2018},
	  month={April},
	}
	
## Requirements
Tested on Ubuntu 16.04 and Windows 10 with `pytorch 1.0`, but should also work with `pytorch 0.4` and Ubuntu 18.04

- `python >= 3.5` and `numpy >= 1.13`
- [`pytorch`](https://pytorch.org/), find installation instructions on [https://pytorch.org/](https://pytorch.org/)
- [`scikit-learn`](https://scikit-learn.org/stable/)
- [`plyfile`](https://github.com/dranjan/python-plyfile) to read point clouds as `.ply` files
- [`pyyaml`](https://pyyaml.org/) to read config files as `.yaml` files

To install `scikit-learn`, `plyfile` and `pyyaml`, you can use:

	sudo pip3 install sklearn plyfile pyyaml

## Installation
To install (it won't download the datasets), just use:

	git clone https://github.com/xroynard/ms_deepvoxscene.git

If you also want to download the pre-processed datasets use instead (**Warning**: datasets may take a lot of space, see below):

	git clone --recursive https://github.com/xroynard/ms_deepvoxscene.git

Size of datasets:

- [Paris-Lille-3D](https://gitlab.com/XavR/parislille3d): takes 5.7 GB on disk
- [Semantic3D reduced-8 benchmark](https://gitlab.com/XavR/semantic3d): takes 5.2 GB on disk
- [S3DIS](https://gitlab.com/XavR/s3dis): takes 5.8 GB on disk

If you already cloned the repository you can either download all datasets with:

	cd ms_deepvoxscene
	git submodule update --init --recursive

Or download a specific datasets with (use _parislille3d_, _semantic3d_ or _s3dis_ as `<dataset_name>`):

	cd ms_deepvoxscene
	git submodule update --init data/<dataset_name>


## Repository Structure
This repository has the following arborescence:

	ms_deepvoxscene ─┬─ apps ─┬─ test.py                                            # scripts for different tasks
                     │        ├─ train.py
	                 │        └─ visualize.py
	                 ├─ config ─┬─ train_config.yaml                                # config files
                     │          ├─ test_config.yaml
                     │          ├─ debug_config.yaml
	                 │          └─ article_configs ─┬─ train_voxnet.yaml            # config files used in article
                     │                              ├─ train_ms1_dvs.yaml
	                 │                              └─ train_ms3_dvs.yaml 
	                 ├─ data ─┬─ parislille3d ─┬─ train                             # point cloud datasets
	                 │        │                └─ test
	                 │        ├─ semantic3d ─┬─ train
	                 │        │              └─ test
	                 │        └─ s3dis ─┬─ train
	                 │                  └─ test
	                 ├─ input ─── input.py                                          # class PointCloudDataset
	                 ├─ models ─┬─ multiscale_models.py                             # multiscale models
                     │          ├─ octree_morton_models.py
                     │          └─ conv_base ─┬─ voxel_models_base.py 
	                 │                        └─ basic_blocks ─── voxel_blocks.py
	                 ├─ runs                                                        # will be created when you run some scripts 
	                 ├─ tests ─┬─ test_input.py                                     # unit tests 
                     │         ├─ test_model_bases.py
                     │         ├─ test_multiscale_models.py
                     │         ├─ test_trainer.py
	                 │         └─ test_voxel_blocks.py
	                 └─ utils ─┬─ logger.py                                         # utils
                               ├─ octree.py
                               ├─ parameters.py
                               ├─ ply_utils.py
                               ├─ tester.py
	                           └─ trainer.py

## Train a Multi-Scale Network
To train a network:

	cd apps
	python3 train.py

You can change parameters in file `config/train_config.yaml` or use your own config file in YAML format with:

	python3 train.py --config <your_config_file>

The script writes all logging informations and saves trained models in `runs/train_<model_name>_<unique_id>/`.

You can check evolution of accuracy and loss during training with:
    
	cd apps
	python3 visualize.py ../runs/<log_repo>/

## Reproduce Article Results 
To reproduce article results train a network with one of the config files in `config/article_configs/`, for exemple to train [`VoxNet`](https://ieeexplore.ieee.org/abstract/document/7353481):

	cd apps
	python3 train.py --config ../config/article_configs/train_voxnet.yaml

## Test and Submit Results
To test a network:

	cd apps
	python3 test.py ../runs/<log_repo>/<trained_model>.tar

You can change parameters in file `config/test_config.yaml`.

## Implementation details

### class `PoinCloudDataset`
This class takes à `.ply` file containing a non-subsampled point cloud and looks for all subsampled point clouds in directories `sub_*cm` lying in the same directory. If there is none, no worries, it will query neighborhoods in the full non-subsampled point cloud (but it might be very time-consuming for big neighborhoods or very dense point clouds).

See [`input/input.py`](input/input.py) for more details.

The class `ConcatenateDataset` of pytorch is then used to iterate over all scenes of a same dataset. 

### class `MultiScaleTemplate`
This class takes a class defining a convolutionnal head (without linear layers and softmax) and a class defining the classification part (only the final linear layers and softmax) and a few parameters such as the number of scales to build a multiscale network.

See [`models/multiscale_models.py`](models/multiscale_models.py) for more details.

## Datasets and Pre-processing
If you don't download the provided datasets in submodules, you should put the datasets in a directory `data` with arborescence:

	data ─┬─ parislille3d ─┬─ train
	      │                └─ test
	      ├─ semantic3d ─┬─ train
	      │              └─ test
	      └─ s3dis ─┬─ train
	                └─ test

each directory `train` or `test` should contain point clouds in `.ply` files and subdirectories named `sub_<subsampling_distance>cm/` where `<subsampling_distance>` is an `int` or a `float` and the subdirectory contains subsampled versions of point clouds.

## License
Our code is released under MIT License (see LICENSE file for details).

## Changelog
- **11/03/2021**: First implememtation of Octree with Morton Code
- **26/06/2019**: Initial Release

## TODO List
- more unit tests.
- improve visualization of training.
- try segmentation networks (like U-Net).
- add option for non-isotropic occupancy grid and voxels
- add a script `utils/make_submission_file.py` that creates the files to be submitted on the Semantic3D and Paris-Lille-3D benchmarks.
