# UNet-ResNet101 for Crack Segmentation

This repository contains an implementation of a **UNet architecture with a ResNet101 encoder** (pretrained on ImageNet) for **semantic segmentation of cracks** in pavement and infrastructure images.  
The main objective is to **detect and localize fine cracks** from high-resolution images where the target objects are very thin and small.

- **Task**: Binary segmentation (crack vs. background)  
- **Dataset**: ~11,000 images at 448×448 resolution  
- **Model**: UNet + ResNet101 (encoder)  
- **Training**: Fine-tuned from pretrained ResNet101 weights  
- **Output**: Pixel-level segmentation masks  

---

## Dataset

The dataset used in this project was obtained from the repository:  
[khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation)  

If you use this dataset in your work, please also consider citing the original source as mentioned in that repository.


- Number of images: ~11k  
- Image size: 448 × 448  
- Labels: Binary masks (0 = background, 1 = crack)  
- Preprocessing:  
  - Normalization with ImageNet mean and std  
  - Data augmentation (random flips, rotations, contrast, brightness, etc.)

---

## Model Architecture
- **Encoder**: ResNet101 pretrained on ImageNet  
- **Decoder**: UNet upsampling path with skip connections  
- **Loss Function**: Binary Cross-Entropy + Dice Loss  
- **Optimizer**: Adam  
- **Metrics**: IoU, Dice Score, Pixel Accuracy  

In this case, the model was trained for 50 epochs with a batch size of 8. My RTX 4060 with 8 Gb of VRAM could barely handle this training.

---

## Results
The model was able to segment cracks, however it also detects some other lines. Results could have been better if the would have been trained with more epochs and higher quality dataset. In the notebook you can see that the model works perfectly on 448 x 448 images.

You can check a video of the inference here:

[Crack segmentation with a U-Net](https://youtu.be/_J0nOdpUgdE)  

You can also download the weights of the model that I trained locally:

[U-Net weights](https://drive.google.com/file/d/1mk1xpo2jEl3y3kzXRxni2qN1Y-2Q_Xg4/view?usp=sharing)  

---

# Citation
Note: the repo of the dataset asks to cite the corresponding papers.

CRACK500:
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}' .

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={arXiv preprint arXiv:1901.06340},
  year={2019}
}

GAPs384: 
>@inproceedings{eisenbach2017how,
  title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
  author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus
          and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
          and Gross, Horst-Michael},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={2039--2047},
  year={2017}
}

CFD: 
>@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

AEL: 
>@article{amhaz2016automatic,
  title={Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection.},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent}
}

cracktree200: 
>@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}

>https://github.com/alexdonchuk/cracks_segmentation_dataset

>https://github.com/yhlleo/DeepCrack

>https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF