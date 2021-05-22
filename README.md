# Image2StyleGAN
* This is a tensorflow 2.x implementation of 'Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?
' (https://arxiv.org/pdf/1904.03189.pdf).

* I would really like to thank @moono for a wonderful StyleGAN2
implementation in tensorflow 2.x (https://github.com/moono/stylegan2-tf-2.x)

Please check the above mentioned repository for details regarding the process
of extracting official pretrained weights.

The pickle file containing the dictionary of latent vectors is stored in the input image folder.

`project.py` is the python file which contains the required code for projection.

Here `data_0` and `data_1`  folders contain images from LSUN (cat dataset), and their corresponding embeddings is stored
in a '.pkl' file, present in the folders.

Google drive link: https://drive.google.com/drive/folders/1IYcIEXV1J0wA_lP1Jbh7qkz2PvX6Dc9S?usp=sharing

* Some results:


| Original Images| Projected Images|
| :---: | :---: |
| ![cat.1.jpg]| ![proj_cat.1.jpg]|
| ![cat.12523.jpg]| ![proj_cat.12523.jpg]|
| ![cat.12592.jpg]| ![proj_cat.12592.jpg]|



[cat.1.jpg]: Image2style_gen/real_cat.1.jpg
[cat.12523.jpg]: Image2style_gen/real_cat.12594.jpg
[cat.12592.jpg]: Image2style_gen/real_cat.12592.jpg
[proj_cat.1.jpg]: Image2style_gen/step_6000_cat.1.jpg
[proj_cat.12523.jpg]: Image2style_gen/step_6000_cat.12594.jpg
[proj_cat.12592.jpg]: Image2style_gen/step_6000_cat.12592.jpg