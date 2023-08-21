---
title: Swapping Autoencoder for Deep Image Manipulation
date: 2022-03-07T09:10:00-04:00  
categories:
-   blog  
tags:
-   image-editing
---

## 논문스터디 : Swapping Autoencoder for Deep Image Manipulation

|이름| **Swapping Autoencoder for Deep Image Manipulation**|
|---|---|
|저자| **Taesung Park**, **Jun-Yan Zhu**, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei A. Efros, **Richard Zhang** |
|토픽| Image Editing |
|년도| 2020.07 |
|학회| NeurIPS 2020|
|링크| [[논문]](https://arxiv.org/abs/2007.00653v2), [[코드1]](https://github.com/taesungp/swapping-autoencoder-pytorch), [[코드2]](https://github.com/rosinality/swapping-autoencoder-pytorch) |

![Image Texture Swapping 예시](/assets/images/swap_ae/swap1.jpeg)

## Intro

**Swapping Autoencoder**는 여타 GAN Inversion / Editing 처럼, 실제 이미지에 대한 수정 을 가능케 합니다.  
이전까지 해당 분야의 많은 논문들은 이미지 수정 시 (의도되었든 의도되지 않았든) *입력 이미지에서만 존재하는 정보*와 *이미지 도메인 전반에 존재하는 정보*를 분리해서 생각해왔고,  
(예시 : Image Cartoonization 실행시, *입력 이미지의 얼굴 구조* 와 *카툰 도메인에서의 텍스쳐 및 구조 정보*),  
이를 위해 대체로 supervised training을 사용해 왔습니다.  
저자들은 이 문제를 **unsupervised training**으로 이 문제를 해결합니다.

해당 논문의 핵심은 AutoEncoder 구조상에서 **이미지를 구조와 텍스쳐**라는, 독립적인 두 요소로 나누어서 Image Encoding을 수행한 후,  
서로 다른 이미지에서 나온 구조와 텍스쳐를 합쳐서 Generation을 거쳐도 실제 이미지와 유사하도록 훈련하는 것에 있습니다.  
(논문에서 구조는 이미지 안에서의 시각적 패턴을, 텍스쳐는 그 외의 정보들을 뜻합니다. 해당 논문을 인용한 다른 논문들에서는 구조 (structure)를 content로, 텍스쳐 (texture)를 style로 쓰기도 하며, 구조와 텍스쳐가 일반적인 관념만큼 잘 떨어지는게 아니기 때문에 해석의 여지가 있습니다.  
(이 논문리뷰에서는 '구조'를 *미시적인 요소들로 구성된 2차원 정보*로, '텍스쳐'는 *거시적인 요소들로 구성된 1차원 정보*로 정의합니다.)


구조와 텍스쳐를 보다 잘 나누기 위해, 저자들은 **이미지 패치 단위로 확인하는 Discriminator**를 사용해서 구조 latent code는 구조를,  
텍스쳐 latent code는 텍스쳐를 encoding 하도록 강제합니다. 

이런 구조를 통해, Swapping Autoencoder **이미지 보간, 이미지 구조 / 텍스쳐 분리 및 수정, 이미지 편집** 등 다양한 이미지 수정 임무를  
기존 유사한 시도들과 비교했을 때 시각적으로도 좋은 결과물을 효율적으로 생산할 수 있음을 보여줍니다.  
또한 저자들은 StyleGAN 같은 unconditional image generation 논문들(과 거기서 파생된 Inversion Methods)과 비교하면서,  
해당 논문들은 사전에 정의된 분포에서만 이미지를 생성하기 때문에 실제 이미지들에 잘 적용되지 않거나, 느리다는 점을 지적합니다.  

### 키워드
* 미시적 구조와 거시적 텍스쳐의 분리된 encoding
* AutoEncoder 구조

---
## 모델 구조

![Model overview](/assets/images/swap_ae/overview.jpeg)

이름에서 보이듯, Swapping Autoencoder의 기본적인 구조는 **Encoder와 Generator, Discriminator로 이루어진 Autoencoder**입니다.  
이 때, Discriminator 디자인은 StyleGAN2에서, Encoder/Generator 블록 구조는 ResNet에서 사용했습니다.

저자들은 3가지 목적을 가지고 해당 모델의 구성 및 훈련 계획을 세웠습니다; 1. 정확한 이미지 재구성, 2. 서로 다른 요소들을 encoding하는 latent code, 3. 이미지 패치 단위 Discriminator를 통한 구조-텍스쳐 분리.  
이중 첫 번째 목표인 이미지 재구성의 경우, 기존과 동일한 reconstruction loss와, Discriminator를 사용한 GAN loss로 해결이 가능합니다.

$$
\mathcal{L}_{\text{rec}}(E,G) = \mathbb{E}_{x \sim \mathbf{X}} [||x - G(E(x))||_1]$$

$$
\mathcal{L}_{\text{GAN, rec}}(E,G,D) = \mathbb{E}_{x \sim \mathbf{X}} [-\text{log}(D(G(E(x))))]
$$

### Latent Code Swapping

Latent Code는 앞서 말했듯이 **구조**($z_s$) 와 **텍스쳐**($z_t$)로 나뉘며, 구조는 (효과적인 구조 인코딩을 위해) 2D latent space를, 텍스쳐는 1D latent space를 사용합니다.  
Encoder 상에서는 4개의 Downsampling ResNet 블록을 거친 후, convolution layer를 통해 $z_s$를 구하는 구간과, convolution layer 와 average pooling layer를 통해 $z_t$를 구하는 구간으로 나뉩니다.  
이런 구조를 통해 텍스쳐 ($z_t$)는 구조적 정보가 결여되어 거시적 텍스쳐 정보만을 인코딩하며, 구조 ($z_s$)는 좁은 receptive field를 통해 미시적 구조 정보를 인코딩하게 됩니다.  

(256 * 256 해상도의 이미지가 있을 때, $z_s \in \mathbb{R}^{16 \times 16 \times 8}$이며 $z_t \in \mathbb{R}^{1 \times 1 \times 2048}$ 입니다.)

이때, 이렇게 분리된 latent code들이 여전히 진짜같은 이미지들을 생성할 수 있음을 입증하기 위해 GAN loss를 추가로 사용합니다. 

$$
\mathcal{L}_{\text{GAN, swap}}(E,G,D) = \mathbb{E}_{x^1, x^2 \sim \mathbf{X}, x^1 \neq x^2} [-\text{log}(D(G(z^1_s, z^2_t)))]
$$

즉, (훈련 시 랜덤하게 뽑힌) 서로 다른 이미지 $x^1, x^2$ 와 거기서 나온 latent code $(z^1_s, z^1_t), (z^2_s, z^2_t)$들이 있을때,  
이미지1의 구조를 encoding한 $z^1_s$과, 이미지2의 텍스쳐를 encoding한 $z^2_t$를 Generator에 넣어도, 실제 같은 이미지가 생성되는지를 확인하는 Loss 입니다.  

기존 GAN이나 VAE처럼 gaussian latent space를 추구하는 다른 논문들과는 달리, 저자는 **latent space의 분포에 어떠한 정해진 분포를 사용하지 않으며**, 다만 개별적인 이미지들의 수정이 말이 되게 이루어지는지만 확인합니다.  
(그러나 필자는 간단한 오토인코더 구조를 쓰는 상황에서 이 말이 의미가 있는지, 그리고 StyleGAN 처럼 w space, s space 처럼 분포를 배우는 것과 비교했을 때 결정적인 이점이 있는지 의문이 듭니다.)

### Patch Discriminator : 패치간 텍스쳐 균일 여부

위의 방식대로 latent code를 나눌 순 있었지만, 나눠진 code들이 각각 structure와 texture를 encoding하기 위해서 저자들은 Patch Discriminator를 새로 추가한다.  
이 Discriminator의 전제는, **텍스쳐가 같은 이미지들의 부분들은 (구조 정보가 누락되어 있으므로) 서로 유사해야 한다는 것**에 있습니다. 

이 전제를 위해 Discriminator는 텍스쳐의 base가 되는 원본 이미지 $x^2$와 그 이미지에서 텍스쳐를 뽑아 새로 만든 변조 이미지 ($G(z^1_s, z^2_t)$) 간의 이미지 패치들을 랜덤하게 선정해서, 그 패치들이 서로 같은 분포에 있는지를 확인합니다.  
이 때, 원본 이미지에서는 여러 장의 패치들을, 변조 이미지에서는 한 장의 패치를 랜덤하게 선택해서, 변조 이미지의 패치가 원본 패치들의 분포에 있는지를 확인하며, 전체 이미지의 크기에 비례해 1/16 ~ 1/64 크기가 되도록 패치를 뽑습니다.  

이러한 텍스쳐에 대한 전제는 Julesz의 텍스쳐 인지론에 영향을 받았다고 저자들은 밝힙니다.

$$
\mathcal{L}_{\text{CooccurGAN}}(E,G,D_{\text{patch}}) = \mathbb{E}_{x^1, x^2 \sim \mathbf{X}} [-\text{log}(D_{\text{patch}}(\text{crop}(G(z^1_s, z^2_t)), \text{crops}(x^2)))]
$$

## 실험 및 결과

저자들은 얼굴 (FFHQ), 동물 (AFHQ), 건축물 (LSUN Churches, Bedrooms), 그림 (Portrait2FFHQ), 자연물 (Flickr Mountain, Waterfall) 등 **다양한 데이터셋**에 대해 실험을 실시했고,  
이 중 그림, 자연물 데이터셋 등은 직접 수집해서 제작했습니다.

비교군으로는 Image2StyleGAN, STROSS, WCT을 선정했습니다.

![reconstr](/assets/images/swap_ae/reconstr.png)

Image Embedding (이미지 $x$ 를 latent code $z$로 바꾸고 reconstruction이 얼마나 잘 되는지) 에서 Image2StyleGAN 과 LPIPS metric으로 비교했을 시,  
LSUN Church를 제외한 데이터들에서 높은 성적을 기록했고 AutoEncoder 구조상 속도도 1000배 가까이 빠른 모습을 보입니다.  
Image2StyleGAN과만 비교했기 때문에 SOTA 급이라고는 할 수 없지만, image reconstruction이 **뿌옇지 않고 미시적인 구조들을 유지**하고 있는 것과,  
빠른 속도로 Image Embedding / Reconstruction이 이루어짐을 볼 수 있습니다. 

![swap2](/assets/images/swap_ae/swap2.jpeg)

두 이미지의 구조와 텍스쳐를 섞어서 이미지를 편집하는 Image Editing의 경우,  
저자들은 Amazon Mechanical Turker (이하 AMT) 들을 통해 설문조사를 실행해서  
Swapping AutoEncoder가 타 모델 대비 더 자연스러운 편집을 해준다는 결과를 확인할 수 있었습니다.  
또 놀라운건 별도의 semantic input이 없이도 하늘, 건물 벽, 지면 등의 요소들이 **일관되게 텍스쳐가 바뀌는** 모습을 확인할 수 있었습니다.  
(하늘의 색깔이 건물에 스며들거나, 건물 텍스쳐가 땅에 입혀지는 일이 거의 없이 알맞게 들어갔습니다.)

![interp1](/assets/images/swap_ae/interp1.gif)
![editing](/assets/images/swap_ae/edit1.gif)

Latent code 에 대한 interpolation을 통한 image translation / editing도 부드럽게 되었습니다.  
가령 그림을 사진으로 바꾸는 태스크에서 저자들은 그림과 사진을 같이 훈련 시킨 후, 그림에 해당하는 데이터들의 latent code와 사진 latent code들의 차이를 계산해서, 그 차이만큼 입력에 적용사키는 걸로 그림에서 사진으로의 image translation을 실행합니다.  
이런 비교적 단순한 코드 적용으로도 자연스러운 image translation이 가능할 뿐만 아니라, 2D 구조 latent code 위에 **그림판으로 그리듯 수정**하는 식으로 image editing도 수행할 수 있음을 보여줍니다.
(사전에 PCA로 뽑은 latent vector들을 추가하는 식으로)

## 정리

Swapping AutoEncoder는 Image Editing task에 있어서 **굳이 Unconditional Image Synthesis --> GAN Inversion을 거칠 필요가 없으며**, 
AutoEncoder 구조로도 효과적인 Image Editing이 가능하다는 것을 보여줍니다.

