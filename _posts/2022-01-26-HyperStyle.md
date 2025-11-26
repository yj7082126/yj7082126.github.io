---
title: Hyperstyle - StyleGAN Inversion with HyperNetworks for Real Image Editing
date: 2022-01-26T15:10:00-04:00  
categories:
-   blog  
tags:
-   stylegan
-   gan-inversion
---

## Hyperstyle (StyleGAN Inversion with HyperNetworks for Real Image Editing)

|이름| **HyperStyle : StyleGAN Inversion with HyperNetworks for Real Image Editing**|
|---|---|
|저자| **Yuval Alauf**, Omer Tov, Ron Mokady, Rinon Gal, Amit H. Bermano |
|토픽| StyleGAN, GAN-Inversion |
|년도| 2021.11 |
|학회| |
|링크| [[논문]](https://arxiv.org/abs/2111.15666), [[코드]](https://github.com/yuval-alaluf/hyperstyle) |

![실사 얼굴 도메인에서의 이미지 복원과 수정.](https://www.linuxadictos.com/wp-content/uploads/HyperStyle-1.jpg)

### Intro
GAN Inversion을 골치 아프게 하는 문제는, **입력 이미지의 복원과 수정간의 균형**을 맞추기 매우 어렵기 때문입니다; 이미지 복원을 중시하며 GAN Inversion을 실행시 사용되는 latent space region들은, 의미론적 수정을 하기 적합한 공간들이 아닌 경우가 매우 많습니다. (출처: [II2s](https://github.com/ZPdesu/II2S))

기존 논문들의 경우, 단일 latent vector를 직접적으로 최적화하는 경우 (예 : [Image2StyleGAN](https://github.com/zaidbhat1234/Image2StyleGAN)) 이미지 복원에서 탁월한 성능을 보이나 수정이 잘 되지 않으며 단일 이미지에 몇분 이상의 시간을 소요하기 때문에 실용적이지 못합니다. 반면 별개로 훈련시킨 Encoder 기반 모델들은 (예 : [pSp](https://github.com/eladrich/pixel2style2pixel)) 더 효율적이나 마찬가지로 이미지 복원과 수정간의 간극을 좁히지 못하는 모습을 보입니다.

최근에 [PTI](https://github.com/danielroich/PTI) 는 기존의 latent vector optimization 대신, StyleGAN 모델을 이미지에 대해 finetune 하여 수정이 용이한 latent vector space로 이미지가 해석되도록 해서 복원/수정간 균형을 맞추었으나, 이 방법도 단일 이미지에 1분가량 걸리기 때문에 **실용적이진 못합니다**.

HyperStyle의 저자들은 바로 이 [PTI](https://github.com/danielroich/PTI)의 구조를 Encoder-based 방법과 합쳐, StyleGAN모델의 weight 값을 배우는 hypernetwork를 훈련시켜 복원/수정간 균형 및 실용성도 잡는 모델을 소개합니다. 단순하게 생각하면 StyleGAN (2 기준) 의 모델 변수 숫자는 3000만개가 넘기 때문에, 저자들은 네트워크 디자인을 통해 Encoder 모델을 효율적으로 디자인해서 모델의 실용성을 보장하고, image translation (도메인 변경)을 비롯한 다양한 태스크에 적용될 수 있음을 보여줍니다.

---

### Previous Works: 
* [33] : [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) ([논문](https://arxiv.org/abs/2006.06676))
    * 지속적으로 커지는 크기의 conv.layer를 통해 이미지를 생성하는 모델 시리즈로, 특유의 latent space와 expressiveness로 이미지 생성 및 GAN inversion에 대표적으로 쓰이는 모델입니다. 본 논문에서는 StyleGAN2 (ada도 포함) 구조를 전제로 설명하고 있습니다.
* [68] : [encoder4editing (aka e4e)](https://github.com/omertov/encoder4editing) ([논문](https://arxiv.org/abs/2102.02766))
    * 이미지 수정에 용이한 encoder-based GAN inversion 모델이며, 준 SOTA급 이미지 복원 능력과 상대적으로 좋은 수정 능력을 가지고 있습니다. 본 논문에서는 latent vector의 생성시 해당 모델을 사용합니다.
* [5] : [Restyle](https://github.com/yuval-alaluf/restyle-encoder) ([논문](https://arxiv.org/abs/2104.02699))
    * 순차적으로 이미지에 대한 latent code optimization을 실행하는 encoder-based GAN inversion 모델이며, 위의 e4e 같은 encoder를 베이스로 사용, 이미지 복원 능력에 SOTA급 능력을 보여줍니다. 본 논문에서도 해당 순차적 구성을 사용합니다.
* [23] : [HyperNetworks](https://github.com/g1910/HyperNetworks) ([논문](https://arxiv.org/abs/1609.09106))
    * 특정 네트워크의 weight들을 예측하는 네트워크로, 해당 네트워크가 특정 입력에 대해 더 expressive한 출력을 낼 수 있도록 해줍니다. 본 논문의 핵심적 역활을 합니다.
* [58] : [PTI](https://github.com/danielroich/PTI) ([논문](https://arxiv.org/abs/2106.05744))
    * latent vector optimization 으로 입력 이미지에 대한 latent vector를 구한 후, 이미지의 수정이 용이하도록 해당 latent vector에 대해 StyleGAN generator를 재훈련시키는 모델입니다. 이미지의 복원과 수정 양쪽에서 좋은 결과를 보여주지만, inference 시간이 느리다는 단점이 있으며, HyperStyle은 바로 이 점을 고치고자 합니다.

***
## Method

### 선행연구

GAN-Inversion 문제를 Latent vector optimization 으로 풀 경우, 
* 목표는 주어진 입력 이미지 / 훈련된 모델 (StyleGAN2를 전제로 함) / 손실 함수 (L2 또는 LPIPS) 가 있을 때, 손실 함수를 최소화 하는 latent vector를 찾는게 됩니다. 
* 이는 보통 단일 이미지에 대해 수분의 시간을 소요하게끔 하며, 따라서 비효율적입니다.

위의 문제를 해결하기 위해 Encoder based optimization 을 사용할 경우,
* 여러 장의 훈련 이미지 데이터셋에 대해 손실함수를 최소화 시키는 방향으로 훈련해, inference시 입력 이미지에 대한 latent vector를 바로 출력하게끔 합니다.
* 이 경우 몇초 정도만 걸리며, 이후 이미지를 수정시 별개의 latent manipulation 을 통해 수정된 latent vector를 StyleGAN2에 입력해서 원하는 수정된 이미지를 얻을 수 있습니다.
* 그러나 latent vector optimization에 비해 이미지 복구는 상대적으로 질이 떨어집니다.

[PTI](https://github.com/danielroich/PTI)는 encoder와는 다르게, StyleGAN2 generator를 수정해서 목적을 달성합니다.
*  입력 이미지에 대한 latent vector optimization을 통해 estimated latent vector를 구하고, 이 latent vector에 대해 손실함수를 최소화 시킬 수 있는 StyleGAN2 weight를 구해, 이미지 복원 / 수정간 균형을 맞추게 합니다 (이때 latent vector는 고정됩니다). 
*  그러나 latent vector optimization이 기본 골자이며, StyleGAN2 weight optimzation 에 대해 걸리는 시간도 길기 때문에 해당 방법은 역시 비효율적 입니다.

***
### Overview

| ![모델 구조](/assets/images/hyperstyle/model-overview.png) | 
|:--:| 
| HyperStyle 모델 구조 : 입력 이미지를 ReStyle-e4e에 넣어 latent vector를 얻은뒤, StyleGAN에 넣어서 복원된 이미지를 얻습니다. 이후 입력 이미지와 복원된 이미지를 HyperNetwork에 넣어, StyleGAN conv layers에 대한 weight offset을 얻습니다. 그리고 latent vector와 weight offset을 적용시키면 (더 잘) 복원된 이미지를 얻게 됩니다.|

저자들이 제시한 HyperStyle은 PTI 에 Encoder-based 방법론을 적용한 후, HyperNetwork를 StyleGAN2 weight optimzation 대신 사용해서 목표를 달성합니다.

우선, estimated latent vector 를 구할 땐 (비교적) 이미지 수정에 용이한 방향으로 latent vector를 출력하는 [e4e](https://github.com/omertov/encoder4editing)을 사용합니다. 
* 초기 e4e 모델의 선택이 중요합니다.
* (당연하지만) latent vector에 대한 별개의 수정은 없기 때문에 기존 e4e에 사용하는 이미지 수정 방법론들을 그대로 사용할 수 있습니다.

latent vector를 구한 후, PTI 처럼 StyleGAN에 대한 **직접적인 optimization 을 하는게 아닌**, 주어진 이미지(들)에 대해 최적의 StyleGAN weight 를 구하는 HyperNetwork H를 사용한 후, 이 weight들과 latent vector를 통해 구한 이미지가 원래 이미지와 같도록 훈련합니다. 
* 이때 HyperNetwork의 입력은 원 입력 이미지와, latent vector를 넣어 StyleGAN에서 나온 이미지 두개를 넣습니다.

엄밀히 말하자면, HyperNetwork는 StyleGAN weight 그 자체를 구하는게 아닌, 입력값에 대한 최적의 weight offset를 구하고, inference 시에는 [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) 의 원리를 본따, **순차적으로 weight offset을 통해 StyleGAN을 수정**합니다.(default : 2~5번)

풀어서 해석하면, HyperStyle은 **단일 이미지 (그리고 단일 latent vector)에 대해 최적의 StyleGAN generator를 배우는거**지만, 일반적인 이미지와는 다르게 ReStyle 와 크게 차이나지 않는 수준으로 inference 속도가 빠르며, 이미지의 복원/수정간 균형도 잘 잡는 모습을 보입니다.

***

### HyperNetwork 에 대하여

| ![모델 구조](/assets/images/hyperstyle/hyperstyle-refinement.png) | 
|:--:| 
| HyperStyle 안 Shared Refinement Block (위) 와 Refinement Block (아래)의 구조들입니다. 둘 다 channel_in * channel_out * 1 * 1 크기의 weight offset을 출력하며, Shared Refinement Block의 경우 공유되는 소 네트워크를 가져 information sharing을 가능케 합니다.|

StyleGAN의 모델 변수 갯수는 약 3천만개가 넘으며, 모든 변수들에 대해서 weight offset을 구하는 hypernetwork는 그 변수가 약 30억개를 넘을 예정입니다. 따라서 효율과 실효성 사이의 균형을 맞추기 위한 네트워크 디자인이 중요해집니다.

우선, HyperNetwork는 ResNet34를 골자로 삼고, 그 목표는 StyleGAN의 convolutional layer들의 weight parameter offset를 배우는걸로 합니다. 

* 여기서 affine transformation layer들의 경우, 어차피 배워야 할 latent vector는 하나이기 때문에 convolutional layer weight를 바꾸는 걸로 affine transformation을 대신할 수 있기 때문에, 굳이 배우지 않습니다. 

* To-RGB 레이어들의 경우, [StyleSpace](https://github.com/betterze/StyleSpace) 논문에 따르면 해당 레이어들을 바꾸는건 pixel-wise texture / color 구조를 바꾸기 때문에, 역시 수정하는 것을 권장하지 않습니다. 

* conv.layer 들 또한, 선택에 따라 중/후반부의 conv. layer들 만을 선택할 수 있고, 모두 선택할 수도 있습니다.

* 따라서 **바꾸는건 (해상도 1024 기준) StyleGAN의 convolutional layer 26개의 conv.weight로 한정되며**, HyperNetwork에는 각 stylegan conv. layer마다 ResNet34의 출력값을 바탕으로 conv.layer 에 대응하는 weight offset을 출력하는 Refinement Block를 둡니다.

HyperNetwork에 있어 가장 중요한 네트워크 디자인 결정은 바로 **channel-wise offset**를 구한다는 겁니다. 

* 예를 들어서, conv.weight 크기가 [512, 512, 3, 3] (in_channel=out_channel=512, kernel_size=3)인 경우, 해당 레이어의 Refinement Block 는 [512, 512, 1, 1] 사이즈의 weight offset을 출력하고, 이를 [3, 3]으로 반복해서 커널에 대해 똑같은 값을 더합니다. 

* 해당 결정으로 88% 가량의 HyperNetwork parameter를 줄일수 있었다고 저자들은 밝히며, 이로 인한 이미지 복원 손상은 발생하지 않았다고 합니다 (다만 channel-wise offset을 해야만 하는 결정적인 이슈는 따로 없는거 같습니다)

여기서 변수들의 숫자를 더 줄이기 위해 **Shared Refinement Block**가 따로 사용됩니다. 
* 다른 Refinement Block들과는 달리 Shared Refinement Block들은 블록의 마지막에 두개의 FC layer들이 있고, 이 레이어들은 모든 Shared Refinement Block에 걸쳐 그 weight가 공유됩니다. 

* 이 특성상 레이어들간의 겹치는 정보를 공유할 수 있고, 필요한 변수들의 숫자를 줄이면서 이미지 복원의 질을 향상시킬 수 있습니다.

* 거시적인 구조들이 초반에 생성되고 미시적인 디테일들이 후반에 잡히는 StyleGAN의 구조상, 해당 Shared Refinement Block들은 초반 conv layer (weight 사이즈가 [512,512,3,3] 인 첫 4개 가량의 레이어들)에 대해서만 사용됩니다.

이 디자인 결정들을 통해, HyperNetwork는 변수들의 숫자를 크게 아껴 inference를 보다 용이하게 합니다. 마지막으로, ReStyle에서 제시한 순차적 refinement를 따와, HyperStyle을 여러번 돌려 최적의 stylegan weight offset을 계산할 수 있도록 합니다.

***

### 훈련 절차

* 데이터셋 : [인간] : FFHQ, CelebA-HQ, [동물, 사물] : Stanford Cars, AFHQ Wild
* 손실함수 : L2 + LPIPS percept.loss + ID Loss
    * 대상이 인간인 경우, ArcFace 기반 ID-loss (pSp에서 쓴것과 동일합니다) 를 사용
    * 대상이 인간이 아닐 경우, MoCo 기반 similarity loss 사용

### 결과 : 이미지 복원

| ![이미지 복원](/assets/images/hyperstyle/hyperstyle-recon.png) | 
|:--:| 
| HyperStyle와 타 모델들의 이미지 복원 능력. direct optimization, PTI와 HyperStyle이 가장 근접한 모습을 보이며, HyperStyle이 가장 선명한 모습을 보입니다.|

HyperStyle은 latent vector optimization 과 동등한 수준의 이미지 복원을 선보이면서, 몇배 빠른 속도를 보여줍니다. 
* 특히 PTI의 경우, 저해상도 이미지의 복원시 이미지에 overfitting하려는 경향 때문에 상대적으로 뿌연 결과를 보여주는 반면, HyperStyle은 Encoder-based 의 latent vector를 사용하면서 그런 해상도에 대한 방해요소가 적습니다. 

pSp이나 e4e같은 인코더 기반 모델과 비교해도 HyperStyle의 성능이 더 좋으며 ReStyle과도 비슷한 수준의 결과를 보여줍니다.

정량적으로 L2-distance, LPIPS-distance, MS-SSIM 같은 수치들을 inference time 대비 비교해봤을떄, HyperStyle의 성능이 타 모델을 크게 뛰어넘는걸 볼 수 있고, 이를 통해 HyperStyle은 latent vector optimzation의 이미지 복원 성능과 encoder-based optimization의 효율성을 동시에 갖췄다고 할 수 있습니다.

|모델|ID|MS-SSIM|LPIPS|L2|Inf.Time(s)|
|---|--|-------|-----|--|--------|
|StyleGAN2|0.78|0.90|0.09|0.020|227.55|
|PTI|0.85|0.92|0.09|0.015|55.715|
|IDInvert|0.18|0.68|0.22|0.061|0.04|
|pSp|0.56|0.76|0.17|0.034|0.106|
|e4e|0.50|0.72|0.20|0.052|0.106|
|ReStyle-pSp|0.66|0.79|0.13|0.030|0.366|
|ReStyle-e4e|0.52|0.74|0.19|0.041|0.366|
|HyperStyle|0.76|0.84|0.09|0.019|1.234|

이미지 복원에 대한 정량적 지표들입니다. HyperStyle은 PTI와 비슷하거나 조금 부족한 수치들을 보이지만 속도에 있어서 수십배 빠르고, 기존 인코더 모델 대비 시간이 조금 느린 대신 더 좋은 수치를 보입니다.
### 결과 : 이미지 수정

| ![이미지 복원](/assets/images/hyperstyle/hyperstyle-edit.png) | 
|:--:| 
| HyperStyle와 타 모델들의 이미지 수정 능력. HyperStyle이 인물, 비인물 가리지 않고 가장 원 identity를 유지하면서 자연스러운 수정이 가해지는 모습을 보입니다.|

이미지 수정 범위의 경우, pSp나 ReStyle-pSp는 latent space에서 수정이 어려운 부분들에 위치한 latent code를 내놓기 때문에 의미있는 수정이 어렵습니다. e4e나 ReStyle-e4e는 수정이 가능한 범위가 좀 넓지만, 원본 대상의 identity preservance가 어렵습니다. 

그러나 HyperStyle은 PTI처럼 latent space에서 수정이 용이한 부분에 위치한 latent code를 제공하기 때문에 원본의 identity를 유지하면서 넓은 폭의 수정이 가능합니다. 

HopeNet, Anycost, CurricularFace 등의 네트워크 등을 통해 얻은 수치들을 다양한 step size를 놓고 성능을 비교해 봤을때, HyperStyle이 시간 대비 효율과 이미지 복원, 이미지 수정 모두에 대해 SOTA급 성능을 보여줌을 확인할 수 있습니다.
* 다양한 방법들로 구한 latent vector에 동일한 step size를 통한 editing을 해봐야 서로 다른 세기로 변할 것이므로, 여러가지 step size에 대해 부드러운 editing이 가능한지를 놓고 비교합니다

### 결과 : Abalation Study
* StyleGAN 네트워크 상의 coarse layer를 제외한 medium-fine layer들만 HyperNetwork의 대상으로 놓고 훈련시켰을시 속도도 빨라지면서 이미지 복원 성능도 유지하는 모습을 보입니다.
    * 그러나 코드 상에선 일단 모든 레이어들에 대해 학습합니다.
* To-RGB 레이어들을 HyperNetwork로 수정하려고 하면 이미지 수정 능력이 현저히 떨어짐을 확인할 수 있습니다.
* ReStyle같은 순차적인 HyperNetwork 개선으로, 이미지 방해요소들을 상당히 제거할 수 있습니다.
* Shared Refinement Block을 써서 레이어간 information sharing을 원할히 할 수 있습니다.
* Separable Convolution (커널 전체에 대해 같은 offset을 내놓는 channel-wise offset이 아닌, 커널의 행 따로 열 따로 offset을 계산후 곱해서 최종 offset을 제공하는 형태로, 이를 통해 속도가 느려지지만 network expressiveness를 키운다)를 사용하는게 크게 성능을 늘려주지 못합니다.

### 결과 : 다른 task에 대해서

| ![도메인 혼합](/assets/images/hyperstyle/hyperstyle-domain.png) | ![도메인 혼합](/assets/images/hyperstyle/hyperstyle-out.png)|
|:--:|:--:|
| Domain Adaption 비교 | Out-of-domain 적용 비교 |

Domain Adaption (예시: 실제 인물사진을 웹툰 도메인의 캐릭터로 바꾸는 태스크) 의 경우, StyleAlign 같은 모델들은 변환 과정에서 머리카락이나 기타 색상의 보존이 되지 않는 모습을 보이지만 HyperStyle의 경우, 해당 디테일들을 보다 잘 유지할 수 있습니다.

변환 과정 : 
* 원 입력 도메인 (예: 인물사진) 에 대해 훈련시킨 source StyleGAN 모델과 그 모델에서부터 원하는 출력 도메인 (예: 웹툰 캐릭터)를 fine-tuning/layer sharing 시킨 target StyleGAN, 그리고 source StyleGAN 에 대해 훈련된 e4e, HyperStyle 모델이 있다고 가정합니다. 
* 입력 이미지를 우선 source-e4e로 넣어서 latent vector를 구하고, source-HyperStyle에 넣어서 weight offset을 구합니다. 
* 그리고 해당 offset을 target StyleGAN에 적용시킨 후에, latent vector를 넣어서 이미지를 구합니다.

또한, 기존에 훈련했던 도메인에서 벗어난 도메인 입력 (예 : 실제 인물사진 모델에 웹툰 캐릭터 인코딩)의 경우에도, 타 모델들과는 달리 HyperStyle은 StyleGAN모델 그 자체를 더 맞게 바꿔주기 때문에, 보다 잘 적용되는 모습을 보입니다.

***

### 결론

HyperStyle은 이미지 복원/수정 간의 균형을 잘 잡아줄 뿐만 아니라, 빠른 시간안에 결과물을 출력해주면서 다양한 입력에 잘 적용되는 모습을 보여줍니다. 해당 논문의 방향성을 잘 연구해서 StyleGAN3 같은 논문에 적용시키거나, 여러 방식으로 훈련을 하면 highly reliable한 GAN-Inversion method이 될 거라고 생각됩니다.