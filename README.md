

<p align="center">

  <h1 align="center">HydraNet</h1>

  <p align="center">
    HydraNet is a segmentation neural network, 
    based UNET architecture with 1 shared encoder and 7 decoders
    <br />
    Ayelet Talby,  Shiri Almog
   
  </p>




<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About](#ABOUT)
* [HydraNet_Main](#built-with)
* [Transfer](#built-with)
* [Contact](#contact)

<!-- ABOUT  -->
## ABOUT
HydraNet is a Unet based neural network, with one encoder and seven different decoders.
We trained HydraNet on 7 different target tasks from the medical domain. The goal was 
to create an encoder that can be usef for transfer learning in the medical field, 
for cases when data is scarce.
<br>
<img src="accessory/HN_logic.PNG" alt="drawing" height="400" width="500"/>


<p>
The medical imaging field suffers from very limited labelled data which makes using 
deep learning for segmentation challenging. In this project we wanted to investigate the 
improvements transfer learning can offer to the research currently being done on 
deep learning in the medical imaging field. Our goal was to create a pre-trained 
network, trained on a diverse multi-organ medical database (MRI and CT) that can 
be used as a base for transfer learning, and to explore the improvement that it 
offers compared to training from scratch or pre-training from ImageNet. 
Our method focuses on creating a deep encoder by using a Fully Convolutional 
Network architecture with one shared encoder and 7 task-related decoders 
(one for each task in our dataset), named HydraNet. Transfer learning from 
the HydraNet encoder to our target data yielded a dice score of 0.7638 while 
training from scratch yielded a dice score of 0.6687, an improvement of 14.22% 
and a difference that has a great medical significance. </p>

## More Detail Coming Soon.... 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation
 
1. Clone the repo
```sh
git clone https://github.com/Shirialmog/HydraNet
```
2. Install PyTorch
```sh
visit https://pytorch.org/
```


 
<!-- CONTACT -->
## Contact

Ayeley Talby - ayelettalby@gmail.com

Shiri Almog - shirialmog1@gmail.com





