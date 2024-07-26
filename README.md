# CSCAM

Code relaese for [Channel-Spatial Support-Query Cross-Attention for Fine-Grained Few-Shot Image Classification](https://arxiv.org/abs/2211.17161). (Accepted in ACM MM-24)

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate CSCAM
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN). 

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- Aircraft \[[Download Link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)\]
- flowers \[[Download Link](https://drive.google.com/file/d/1G4QRcRZ_s57giew6wgnxemwWRDb-3h5P/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]


## Train

* To train FRN+CSCAM on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/FRN_CSCAM/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/FRN_CSCAM/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/FRN_CSCAM/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/FRN_CSCAM/ResNet-12
    python ./test.py
```

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Ma](https://github.com/xmu-xiaoma666/External-Attention-pytorch) and  [Lee](https://github.com/leesb7426/cvpr2022-task-discrepancy-maximization-for-fine-grained-few-shot-classification), for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- shicheng_yang@126.com
