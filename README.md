## Citation
Please cite our work if you think it is useful for your research.
### BibTeX
```
@article{zhang2024rgbt,
  title={RGBT tracking via frequency-aware feature enhancement and unidirectional mixed attention},
  author={Zhang, Jianming and Yang, Jing and Liu, Zikang and Wang, Jin},
  journal={Neurocomputing},
  pages={128908},
  year={2024},
  publisher={Elsevier}
}
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
Dowmload the pretrained [foundation model](https://pan.baidu.com/s/1JX7xUlr-XutcsDsOeATU1A?pwd=4lvo) (OSTrack) (Baidu Driver: 4lvo) / [foundation model](https://drive.google.com/file/d/1WSkrdJu3OEBekoRz8qnDpnvEXhdr7Oec/view?usp=sharing) (Google Drive)
and put it under ./pretrained/.
```
bash train_bat.sh
```
You can train models with various modalities and variants by modifying ```train_bat.sh```.

### Testing

#### For RGB-T benchmarks
[LasHeR, RGBT210 & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.

The project visualization video was published on:https://github.com/jingyang-csust/Demo_video_for_FFE-BMFF

