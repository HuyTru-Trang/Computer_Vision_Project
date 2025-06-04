## Original Paper
Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on. arXiv preprint arXiv:2403.01779, 2024. https://arxiv.org/abs/2403.01779
```
@article{xu2024ootdiffusion,
  title={OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on},
  author={Xu, Yuhao and Gu, Tao and Chen, Weifeng and Chen, Chengcai},
  journal={arXiv preprint arXiv:2403.01779},
  year={2024}
}
```

## Installation
```
!pip install -r requirements.txt
!pip install huggingface_hub==0.20.3
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Getting weights
Download these and put them into checkpoints
* Download checkpoints from weights of paper from https://huggingface.co/levihsu/OOTDiffusion/tree/main/checkpoints
* Download Clip from https://huggingface.co/openai/clip-vit-large-patch14

## Dataset
* Download dataset from https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view and put under /Computer_Vision_Project
* Replace the train_pairs.txt with the one we provide.

## Fine-tuning
```
accelerate launch ootd_train.py --load_height 512 --load_width 384 --dataset_list 'train_pairs.txt' --dataset_mode 'train' --batch_size 1 --train_batch_size 1 --num_train_epochs 15
```
The new weights will be in /Computer_Vision_Project/run/ootd_train_checkpoints

## Try-On
```sh
cd OOTDiffusion/run
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --scale 2.0 --sample 4
```

## Citation
```
@article{xu2024ootdiffusion,
  title={OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on},
  author={Xu, Yuhao and Gu, Tao and Chen, Weifeng and Chen, Chengcai},
  journal={arXiv preprint arXiv:2403.01779},
  year={2024}
}
```

## Acknowledgement
We would like to sincerely thank [lyc092](https://github.com/lyc0929) for having referenced the source at https://github.com/lyc0929/OOTDiffusion-train to rebuild our training code.