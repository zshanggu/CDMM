

-------
&nbsp;

## Installation

### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- Ubuntu LTS 18.04
- 8x NVIDIA V100 GPUs (32GB)
- CUDA 10.2
- Python == 3.7
- PyTorch == 1.7.1+cu102, TorchVision == 0.8.2+cu102
- GCC == 7.5.0
- cython, pycocotools, tqdm, scipy

We recommend using the exact setups above. However, other environments (Linux, Python>=3.7, CUDA>=9.2, GCC>=5.4, PyTorch>=1.5.1, TorchVision>=0.6.1) should also work properly.

&nbsp;

### Code Installation

First, clone the repository locally:
```shell
cd model/ops/
python setup.py build develop
```


&nbsp;

### Data Preparation

#### MS-COCO for Few-Shot Object Detection

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    ├── coco_fewshot/        # Few-shot dataset 
    └── coco/                # MS-COCO dataset
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
```



#### Pascal VOC for Few-Shot Object Detection

We transform the original Pascal VOC dataset format into MS-COCO format for parsing.

After downloading MS-COCO-style Pascal VOC, please organize them as following:

```
code_root/
└── data/
    ├── voc_fewshot_split1/     # VOC Few-shot dataset
    ├── voc_fewshot_split2/     # VOC Few-shot dataset
    ├── voc_fewshot_split3/     # VOC Few-shot dataset
    └── voc/                    # MS-COCO-Style Pascal VOC dataset
        ├── images/
        └── annotations/
            ├── xxxxx.json
            ├── yyyyy.json
            └── zzzzz.json
            └── rich_text.json
```

Similarly, the few-shot datasets for Pascal VOC are also provided in this repo ([`voc_fewshot_split1`](data/voc_fewshot_split1), [`voc_fewshot_split2`](data/voc_fewshot_split2), and [`voc_fewshot_split3`](data/voc_fewshot_split3)). For each class split, there are 10 data setups with different random seeds. In each K-shot (K=1,2,3,5,10) data setup, we ensure that there are exactly K object instances for each novel class. The numbers of base-class object instances vary.

----------
&nbsp;

## Usage

### To Perform _**Pre-training & Finetuning**_

```bash
bash scripts/fsfinetune_coco.sh
```
All training scripts follow this template. Note that you need to add `--fewshot_finetune` to indicate that the training and inference should be conducted on few-shot setups. You also need to specify the number of shots, few-shot random seed, training epoch setups, and the checkpoint file path after base training.
Then, run the commands below to start few-shot finetuning. After finetuning, the program will automatically perform inference on novel classes.
