# Multimodal Emotion Recognition Calibration in Conversations

> The official implementation for paper: *Multimodal Emotion Recognition Calibration in Conversations*, MM '24.

<img src="https://img.shields.io/badge/Venue-ACM MM-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">


## Requirements
* Python 3.10.13
* PyTorch 1.13.1
* torch_geometric 2.4.0
* torch-scatter 2.1.0
* torch-sparse 0.5.15
* CUDA 11.7

## Preparation
1. Download  [**multimodal-features**](https://www.dropbox.com/scl/fo/veblbniqjrp3iv3fs3z6p/AEzkNgWqPHHzldBZ0zEzr2Y?rlkey=yhlr653c0vnvaf1krpdkla36u&e=1&dl=0) 
2. Save data/iemocap/iemocap_features_roberta.pkl, data/iemocap/IEMOCAP_features.pkl in `data/`; Save meld_features_roberta.pkl, data/meld/MELD_features_raw1.pkl in `data/`. 
3. Download IEMOCAP_diff.pkl, MELD_diff.pkl from [google drive](https://drive.google.com/drive/folders/1ty0XonRQG-DOyNLASHd9bkothMlBwjI8?usp=sharing), and put them in `caldiff/`

## Training

### Training on IEMOCAP

1. Train M³Net model for the ERC task using the IEMOCAP dataset.
Please refer to [M³Net](https://github.com/feiyuchen7/M3NET)

```shell
python train_cm.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type 'hyper' \
      --epochs 80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT'\
      --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L 3 --num_K 4 --seed 1475 \
```

2. Calculate the difficulty of each conversation.
```shell
python caldiff/cal_diff.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type 'hyper' \
      --epochs 80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT'\
      --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L 3 --num_K 4 --seed 1475 \
      --ckpt_path='YOUR_M3NET_MODEL'
```

3. Train CMERC on the M³Net model for the ERC task using the IEMOCAP dataset.
```shell
python train_cm.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type 'hyper' \
      --epochs 80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT'\
      --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L 3 --num_K 4 --seed 1475 \
      --calibrate --rank_coff 0.005 \
      --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 \
      --courselearning --epoch_ratio 0.15 --scheduler_steps 1
```


### Training on MELD
1. Train M³Net model for the ERC task using the MELD dataset.
Please refer to [M³Net](https://github.com/feiyuchen7/M3NET)
```shell
python -u train_cm.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=40 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
      --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --seed 67137
```

2. Calculate the difficulty of each conversation.
```shell
python -u train_cm.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=40 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
      --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --seed 67137 \
      --ckpt_path='YOUR_M3NET_MODEL'
```


3. Train CMERC on the M³Net model for the ERC task using the MELD dataset.
```shell
python -u train_cm.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=40 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
      --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --seed 67137 \
      --calibrate --rank_coff 0.002 \
      --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 \
      --courselearning --epoch_ratio 0.4 --scheduler_steps 1
```

### Quick Start
We have already provided the difficulty for each conversation in `caldiff/` directory, so we can skip steps 1-2 mentioned above and use our provided checkpoint for efficient training.

- For IEMOCAP:
```shell
python train_cm.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type 'hyper' \
      --epochs 80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT'\
      --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L 3 --num_K 4 --seed 1475 \
      --calibrate --rank_coff 0.005 \
      --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 \
      --courselearning --epoch_ratio 0.15 --scheduler_steps 1
```

- For MELD:
```shell
python -u train_cm.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=40 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
      --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --seed 67137 \
      --calibrate --rank_coff 0.002 \
      --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 \
      --courselearning --epoch_ratio 0.4 --scheduler_steps 1
```




## Evaluation
1. Evaluation CMERC on the M³Net model for the ERC task using the IEMOCAP dataset.
```shell
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=0 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --testing
```

2. Evaluation CMERC on the M³Net model for the ERC task using the MELD dataset.
```shell
python -u train.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=0 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --testing
```


## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{tu2024calibrate,
title = {Multimodal Emotion Recognition Calibration in Conversations},
author = {Tu, Geng and Xiong, Feng and Liang, Bin and Wang, Hui and Zeng, Xi and and Xu, Ruifeng},
booktitle = {Proceedings of the 32st ACM International Conference on Multimedia},
series = {MM '24}
}
```


## Acknowledgements
Special thanks to the following authors for their contributions through open-source implementations.
- [Emotion Recognition in Conversations](https://github.com/declare-lab/conv-emotion)
- [Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation](https://github.com/feiyuchen7/M3NET)

