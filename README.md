## Citation
Please cite our work if you think it is useful for your research.
### BibTeX
```
@article{ZHANG2025128908,
title = {RGBT tracking via frequency-aware feature enhancement and unidirectional mixed attention},
journal = {Neurocomputing},
volume = {616},
pages = {128908},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128908},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224016795},
author = {Jianming Zhang and Jing Yang and Zikang Liu and Jin Wang}
```
### Word
```
Jianming Zhang, Jing Yang, Zikang Liu, Jin Wang. RGBT tracking via frequency-aware feature enhancement and unidirectional mixed attention. Neurocomputing, 2025, vol. 616, 128908. DOI: 10.1016/j.neucom.2024.128908.
```
## Usage
### Installation
Create and activate a conda environment:
```
conda create -n xxx python=3.7
conda activate xxx
```
Install the required packages:
```
bash install_bat.sh
```

### Data Preparation
Download the training datasets, It should look like:
```
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        ...

    -- RGBT234/TrainingSet
        ...

    -- RGBT210/TrainingSet
        ...
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_BAT>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://pan.baidu.com/s/1rgr-5VSCxhbMYhgwToJ-kg?pwd=nq8j) (OSTrack) (Baidu Driver: nq8j)
and set path.
```
bash train_bat.sh
```
You can train models with various modalities and variants by modifying ```train_bat.sh```.

### Testing
#### Model
[Model](https://pan.baidu.com/s/138-V7ApR4WV1OyhkCHaCCQ?pwd=u5fq) (Baidu Driver: u5fq)
#### For RGB-T benchmarks
[LasHeR, RGBT210 & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.

The project visualization videos are published on:https://github.com/jingyang-csust/Demo_video_for_FFE-BMFF

