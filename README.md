# Adversarial Learning with Mask Reconstruction for Text-Guided Image Inpainting (ALMR) 
## Abstract
Text-guided image inpainting aims to complete the corrupted patches coherent with both visual and textual context. On one hand, existing works focus on surrounding pixels of the corrupted patches without considering the objects in the image, resulting in the characteristics of objects described in text being painted on non-object regions. On the other hand, the redundant information in text may distract the generation of objects of interest in the restored image. In this paper, we propose an adversarial learning framework with mask reconstruction for image inpainting with textual guidance, which consists of a two-stage generator and dual discriminators. The twostage generator aims to restore coarse-grained and fine-grained images, respectively. In particular, we devise a dual-attention module to incorporate the word-level and sentence-level textual features as guidance on generating the coarse-grained and finegrained details in the two stages. Furthermore, we design a mask reconstruction module to penalize the restoration of the objects of interest with the given textual descriptions about the objects. For adversarial training, we exploit global and local discriminators for the whole image and corrupted patches, respectively. 
## Datasets
We use [CUB-200-211](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Oxford-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [CelebaA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ) datasets. 
##  Training
* pre-train: python pretrain.py
* birds:  https://pan.baidu.com/s/1o_rfXUt4JFLZqTNPSrRetQ    qxx8
* celeba15000: https://pan.baidu.com/s/18n-kaFq39H5HFyPy4V2gzA    ohws
* flowers: https://pan.baidu.com/s/1QPDAjzO_1fNlQmUOeOQ-LA    x5pl
* 
* train: python train.py
##  Testing
* test: python test.py
* FID and KID: python fid_kid_all.py
* PSNR and SSIM: python PSNR-SSIM.py
