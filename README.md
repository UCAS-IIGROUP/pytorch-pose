# PyTorch-Pose

PyTorch-Pose is a PyTorch implementation of the general pipeline for 2D single human pose estimation. The aim is to provide the interface of the training/inference/evaluation, and the dataloader with various data augmentation options for the most popular human pose databases (e.g., [the MPII human pose](http://human-pose.mpi-inf.mpg.de), [LSP](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) and [FLIC](http://bensapp.github.io/flic-dataset.html)).

Some codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train). Thanks to the original author. 

## Models 
| Model|in_res |featrues| # of Weights |Head|Shoulder|	Elbow|	Wrist|	Hip	|Knee|	Ankle|	Mean|
| --- |---| ----|----------- | ----| ----| ---| ---| ---| ---| ---| ---|
| hg_s2_b1|256|128|6.73m| 95.74| 94.51| 87.68| 81.70| 87.81| 80.88 |76.83| 86.58
| hg_s2_b1_mobile|256|128|2.31m|95.80|  93.61| 85.50| 79.63| 86.13| 77.82| 73.62|  84.69|
| hg_s2_b1_tiny|192|128|2.31m|95.02|  92.66| 84.37| 76.92| 84.16| 77.13| 72.08|  83.34|


## Installation
1. Create a virtualenv
   ```
   virtualenv -p /usr/bin/python2.7 posevenv
   ```
2. Install all dependencies in virtualenv
    ```
    source posevenv/bin/activate
    pip install -r requirements.txt
    ```
3. Clone the repository with submodule
   ```
   git clone --recursive https://github.com/yuanyuanli85/pytorch-pose.git
   ```

4. Create a symbolic link to the `images` directory of the MPII dataset:
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR data/mpii/images
   ```

5. Disable cudnn for batchnorm layer to solve bug in pytorch0.4.0
    ```
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ./pose_venv/lib/python2.7/site-packages/torch/nn/functional.py
    ```
## Training

* Normal network configuration, in_res 256, features 256
```python 
python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1/ --in_res 256 --features 256
```

* Mobile network configuration, in_res 256, features 256
```python 
python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1_mobile/ --mobile True --in_res 256 --features 256
```

* Tiny network configuration, in_res 192, features 128
```python 
python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1_tiny/ --mobile True --in_res 192 --features 128
```

## Evaluation

Run evaluation to generate mat file
```python
python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/hg_s2_b1/ --resume checkpoint/hg_s2_b1/model_best.pth.tar -e
```
* `--resume_checkpoint` is the checkpoint want to evaluate

Run `evaluation/eval_PCKh.py` to get val score 

## Export pytorch checkpoint to onnx 
```python
python tools/mpii_export_to_onxx.py -a hg -s 2 -b 1 --num-classes 16 --mobile True --in_res 256  --checkpoint checkpoint/model_best.pth.tar 
--out_onnx checkpointmodel_best.onnx 
```
Here 
* `--checkpoint` is the checkpoint want to export 
* `--out_onnx` is the exported onnx file



