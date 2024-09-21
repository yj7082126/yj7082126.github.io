---
layout: post
title: E4E - Encoder for StyleGAN Image Manipulation
date: 2022-02-20T15:10:00-04:00  
categories:
-   blog  
tags:
-   stylegan
-   gan-inversion
---

## 논문스터디 : E4E (Designing an Encoder for StyleGAN Image Manipulation)

|이름| **Designing an Encoder for StyleGAN Image Manipulation**|
|---|---|
|저자| **Omer Tov**, Yuval Alaluf, Yotam Nitzan, Or Patashnik, Daniel Cohen-Or |
|토픽| StyleGAN, GAN-Inversion |
|년도| 2021.02 |
|학회| SIGGRAPH 2021 |
|링크| [[논문]](https://arxiv.org/abs/2102.02766), [[코드]](https://github.com/omertov/encoder4editing) |

![Encoder4Editing 예시](/assets/images/e4e/teaser.jpeg)

## Introduction

GAN 이후로 발표된 여러 이미지 생성모델 중에서도 StyleGAN이 두각을 보인 까닭은, 단순히 생성된 이미지의 질이 좋다는 이유 뿐만이 아닌, 학습을 통해 배운 $\mathcal{W}$ latent space를 사용하는데 있습니다. 

StyleGAN (1,2,3 모두)는 가우시안 latent space $\mathcal{Z}$에서 이미지 분포에 가깝도록 훈련된 latent space $\mathcal{W}$ 로 변형시켜주는 *mapping network*와, $w \in \mathcal{W}$ 에서 이미지를 생성하는, 점진적으로 출력 해상도가 증가하는 형태의 네트워크인 *synthesis network*로 나뉩니다. (이 때, $w$는 synthetic network의 각 레이어마다 들어가는 style vector로 구성되어, $w$의 dimensionality는 $w \in \mathbb{R}^{\text{레이어 수} \times 512}$ 입니다.)

여기서 $\mathcal{W}$ latent space는 다른 모델들의 비교적 단순한 분포들 (예 : 가우시안) 가진 latent space과는 달리, 이미지의 실분포를 보다 잘 설명해주며 latent space 상의 요소들이 실제 이미지 요소에 따라 잘 나뉘어져 있어 (예 : 얼굴에 대한 StyleGAN의 경우 표정, 머리카락, 나이 등) 편집이 용이하다는 장점이 있습니다.

그러나 이런 StyleGAN의 장점을 유지한 채로 실제 이미지에 편집을 하기 위해선, 그 이미지에 대응하는 StyleGAN에서의 latent code $w \in \mathcal{W}$ 를 찾아야 합니다. 이를 *GAN-Inversion*이라고 부릅니다. 이미지에 맞는 $w$를 찾은 후에는 latent space의 성질을 사용해 $w$를 조작해, 이미지를 목표에 따라 편집합니다. 이를 *latent-space manipulation*이라고 부릅니다.

보통 $w$를 구하는 데 사용하는 두가지 방법인 *learning-based GAN Inversion* (StyleGAN과는 별도의 Encoder network를 사용해서 $w$를 계산) 과 *optimization-based GAN Inversion* (별도의 네트워크 없이 $w$ 자체를 최적화해서 계산) 중, 저자들은 전자에 집중하고 있습니다.

## Distortion-Editability, Distortion-Perception Tradeoffs

GAN-Inversion을 통해 구한 $w$는 *1. StyleGAN에 넣었을 때 생성될 이미지가 원래 이미지와 최대한 같아야 하고 (재구성 능력, reconstruction), 2. latent space manipulation 편집을 용이하게 하기 위해 latent space 상에서 분리된 요소를 잘 따라야 합니다. (편집 능력, editability)* 아무리 재구성 능력이 좋아 원래 이미지와 똑같은 이미지를 만들 수 있어도, 특정 목적에 따른 편집이 되지 않으면 이미지 편집을 위해 StyleGAN을 사용하는 의미가 없습니다. 마찬가지로, 아무리 편집 능력이 좋아도 기본적인 이미지 재구성이 되지 않아 원래 이미지처럼 만들 수 없는 경우, 역시 StyleGAN을 사용하는 이미지가 없습니다.

여기서 전자 (재구성)의 경우, 2가지로 나뉘어서 *원 이미지와 생성된 이미지 간의 차이*를 **distortion**, *생성된 이미지가 얼마나 실제 이미지 같은지*를 **perceptual quality**로 정의합니다. 언듯 같은 목표에 대한 개념들로 보이지만, 다음의 예시처럼 두 개념은 상충될 수 있기 때문에, 나누어서 생각하는 것이 중요합니다. 

|![figure5](/assets/images/e4e/figure5.png)|
|-|
|distortion과 perceptual quality간의 trade-off 예시. 가운데 이미지는 입력 이미지 (왼쪽)와 낮은 픽셀 차이 (낮은 distortion)을 보이지만, 말의 사진인데도 눈이나 사람들이 "말"하면 생각날 기본적인 요소들이 결락되어, 낮은 perceptual quality를 보입니다. 반면 오른쪽 이미지는 더 높은 distortion을 보이지만, 실제 말을 찍은 것과 같은 높은 perceptual quality를 보입니다.|

저자들은 이 **3가지 요소 사이에 불균형이 존재**하며, 그 균형을 맞추기는 어렵다고 주장합니다. 

가령 Image2StyleGAN의 경우, 이미지 재구성 능력을 키우기 위해 $\mathcal{W}$의 확장 영역인 $\mathcal{W}+$을 사용하여, 굉장히 낮은 distortion을 보여줍니다. 그러나, 그 과정에서 이미지 편집이 용이하지 않은 latent space 영역으로 $w$의 분포가 이동하면서 이미지 편집이 부자연스럽게 되며, perceptual quality또한 손상되는 문제가 발생합니다. 

## E4E : Encoder for Editing.

|![figure5](/assets/images/e4e/figure6.png)|
|-|
|E4E의 전반적인 구성. $\mathcal{W}+$ 차원의 Encoder 모델은 이미지를 입력받아, 단일 style vector $w$와, offset $\Delta_1, ..., \Delta_{N-1}$ 를 output으로 출력하고, $w$를 N번 반복해서 offset들을 더하는걸로 최종 inversion latent code를 완성합니다.|

이렇게 distortion을 낮추기 위해 perceptual quality와 editability를 희생해야 하는 상황을 피하기 위해, 저자들은 $w$는 최대한 $\mathcal{W}$ 분포와 '가까워'야 한다고 주장합니다. 여기서 '가깝다'라는 것은, *$w$를 구성하는 각 style vector간 variance가 낮으면서*, 그 style vector들이 *$\mathcal{W}+$ 가 아닌 $\mathcal{W}$의 분포에 있음*을 뜻합니다.

이 법칙에 대해 훈련되어 이미지 재구성 및 편집간 균형을 맞추어, 기존보다 뛰어난 성능을 보이는 Encoder network를 저자들은 *Encoder for Editing - 이하 e4e*라고 부릅니다. 모델의 기본적인 구성은 pSp와 유사하지만, 훈련 방식 및 로스 등에서 e4e는 pSp와 큰 차이를 보입니다.

우선 style vector간 variance를 낮추기 위해, 저자들은 새로운 *"progressive training"* 방식을 제안합니다. 기존의 encoder-based optimization 방법들은 모두, 이미지 입력 $x$를 받아 $w_0, w_1, ...$의 서로 다른 style vector들을 생성하는 방식이었고, 이는 style vector들 간의 분포가 지나치게 넓어지는 원인이 되었습니다.

$$E(x) = w \quad (w = (w_0, w_1, ..., w_{N-1}), w \in \mathbb{R}^{N\times512})$$

반대로 e4e는, 한 개의 style vector인 $w_0 \in \mathbb{R}^{512}$와, $w_0$에 더할 (레이어 수-1 개의) **offset** 들을 예측하고, 이 offset들을 각각 적용하는 방식으로 $w$를 생성합니다.

$$E(x) = (w_0, \Delta) \quad (\Delta = (\Delta_1, ..., \Delta_{N-1}), \Delta \in \mathbb{R}^{(N-1)\times512})$$
$$w = (w_0, w_0 + \Delta_1, ..., w_0 + \Delta_{N-1})$$

훈련 시작 시, offset들은 $\forall i : \Delta_i = 0$로 초기화하고 encoder는 $w_0$ 만을 배우도록 훈련한 뒤, 일정 수준 이상의 이미지 재구성 능력이 확보되었을 때, 각 i에 대한 $\Delta_i$를 *순차적으로* (한 차례에 하나씩) 배웁니다. 이를 통해 모델을 초기 레이어의 거시적 이미지 구조부터 시작하여 후기 레이어의 미시적 이미지 요소들을 완성시키는 offset들을 배워 생성된 이미지의 질을 높이면서, style vector들간 variance가 높지 않도록 조절합니다. 특히 variance의 경우, 명시적으로 조절하기 위해 다음의 loss term을 사용합니다.

$$\mathcal{L}_{d-reg}(w) = \sum^{N-1}_{i=1} || \Delta_i ||_2$$

여기에 더해 style vector들이 $\mathcal{W}$의 분포안에 있게끔 하기 위해 Discriminator를 사용, $\mathcal{W}$의 분포에서 뽑은 '진짜' $w$ 들과 Encoder에서 나온 '가짜' $w$들간 구별을 하도록 합니다. 이 Discriminator를 통해 직접적으로 정의 내릴수 없는 $\mathcal{W}$ 공간에 대해 작업을 할 수 있습니다. Loss는, R1 Regularization을 더한 Non-saturating GAN loss를 사용합니다.

$$\mathcal{L}^{D}_{adv} = - \mathbb{E}_{w \sim \mathcal{W}}[\text{log }D_{\mathcal{W}}(w)] - \mathbb{E}_{x \sim p_x}[\text{log }(1 - D_{\mathcal{W}}(E(x)_i))] + \frac{\gamma}{2}\mathbb{E}_{w \sim \mathcal{W}}[||\nabla_wD_{\mathcal{W}}(w)||^2_2]$$
$$\mathcal{L}^{E}_{adv} = -\mathbb{E}_{x \sim p_x}[\text{log }D_{\mathcal{W}}(E(x)_i)]$$

이렇게 $w$ 간 variance를 억제하는 $\mathcal{L}_{d-reg}$와, $\mathcal{W}$의 분포에 오게끔 하는 $\mathcal{L}_{adv}$ 로스를 합쳐서, perceptual quality와 editability를 보존하는 $\mathcal{L}_{edit}$을 사용합니다.

$$\mathcal{L}_{edit} = \lambda_{d-reg}\mathcal{L}_{d-reg} + \lambda_{adv}\mathcal{L}_{adv}$$

한편, distortion (과 perceptual quality의 강화)를 위해선, MoCov2 로 훈련된 ResNet50 네트워크 ($C$)를 사용, 실제 사진과 Encoder를 통해 구한 latent code를 재구성한 가짜 사진들의 ResNet50 feature들간 cosine similarity를 최소화 시키는 loss term을 사용합니다. (대상이 사람의 얼굴일 경우, ArcFace facial recognition network를 사용합니다.)

$$\mathcal{L}_{sim} = 1 - <C(x), C(G(E(x)))>$$

여기에 L2 loss, LPIPS loss 등 distortion / perception quality를 강화하기 위한 로스들을 사용해서, 최종적으로 다음과 같이 로스를 완성시킵니다.

$$\mathcal{L}_{dist} = \lambda_{l2}\mathcal{L}_{l2} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{sim}\mathcal{L}_{sim}$$

$$\mathcal{L} = \mathcal{L}_{dist} + \lambda_{edit}\lambda_{edit}$$