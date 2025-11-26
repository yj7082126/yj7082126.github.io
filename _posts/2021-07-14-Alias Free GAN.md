---
title: StyleGAN3 - Alias Free StyleGAN
date: 2021-02-28T15:10:00-04:00  
categories:
-   blog  
tags:
-   computer vision
-   StyleGAN
---

## StyleGAN3 (Alias-Free Generative Adversarial Networks)

|이름| **Alias-Free Generative Adversarial Networks**|
|---|---|
|저자| **Tero Karras**, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaako Lehtinen, Timo Aila |
|토픽| StyleGAN |
|년도| 2021.06 |
|학회| Proc.NeurIPS|
|링크| [[논문]](https://arxiv.org/abs/2106.12423), [[코드]](https://github.com/NVlabs/stylegan3) |


[![Two Minute Papers-StyleGAN3](https://img.youtube.com/vi/0zaGYLPj4Kk/0.jpg)](https://www.youtube.com/watch?v=0zaGYLPj4Kk&t=167s) 

StyleGAN2-ada 에서 새로이 발전한 이 모델은, 기존 StyleGAN 모델들이 정립했던 여러 방침들에 대해 과감하게 배제하는 선택을-물론 기본적인 StyleGAN의 전제는 바뀌지 않았습니다-내린 후, 그 자리에 신호처리라는 기존에 크게 주목되지 않았던 영역을 도입했습니다.  

이를 통해 해당 논문은 기존의 이미지 생성 퀄러티를 유지하면서, Translational / Rotational Equivariance를 지키고, 나아가 Computer Vision 쪽에서 새로 눈여겨볼만한 연구주제들이 제시되는 배경이 되고 있습니다.

앞으로 생성 모델의 품질 개선부터 영상 단위의 생성모델까지, 해당 모델과 논문에서 제시된 개념들이 응용될 가능성이 무궁무진하다고 생각됩니다.

### Intro
GAN은 실제 환경처럼 **거시적인 구조 (예 : 얼굴)가 미시적인 구조 (예 : 주근깨)를 결정**짓지만, 그 **미시적인 구조의 움직임을 통제하진 못합니다.**
* 예를 들어 StyleGAN으로 생성된 모델을 latent vector 조작을 통해 이동/회전시, 미시적인 구조가 따라 움직이지 못하고 마치 화면에 붙은것 같은 모습을 보입니다.
* 이를 "texture sticking"이라고 하며, 자연스러운 이미지 생성을 저해하는 대표적인 현상입니다.
* 해당 현상의 주 요인은 네트워크 사이 레이어에서 위치 정보가 "세어 들어가는" 현상, StyleGAN에서의 noise input, positional encoding, 그리고 얼라이어싱이 지목되고 있습니다.

저자들은 퀄러티 저해의 주요 요인인 얼라이어싱이 GAN 분야에서 잘 다뤄지지 않는점에 주목하고, 해당 문제를 다음과 같이 해석해서 해결하고자 합니다.
* (저자들에 의하면) 얼라이어싱은 이상적이지 못한 **Upsampling layer** (nearest, bilinear, 기타 등등)과 ReLU 같은 **nonlinearity layer** 에서 주로 발생합니다.
  * 해당 레이어들은 훈련과정에서 다음 네트워크 레이어로 의도되지 않은 정보들을 보내고, 이는 레이어들 안에서 증폭되어 최종 결과에 얼라이어싱으로 크게 반영됩니다.

이 문제를 해결하기 위해선 해당 레이어들과 네트워크 구조 전체를 바꿔야 한다는 것이 저자들의 중론입니다.
* 이 과정에서 **신호처리 관점을 도입**, 기존의 네트워크 구조에 대해 고찰한 후, Translational / Rotational equivariance를 지키도록 수정합니다.
* 특히 Upsampling / Nonlinearity 구조를 원점에서 고려합니다.

최종적으로 완성된 StyleGAN은 실제 환경처럼 거시적/미시적 구조들의 움직임을 완벽에 가깝게 통제, 기존의 aliasing 문제를 상당히 해결합니다.
* 이를 통해 StyleGAN 내부 latent space에 대한 새로운 해석의 관점을 도입하고, 향후 영상 단위의 생성에 대한 여지를 남깁니다.

***

### Previous Works

* [49] : Shannon 저. [Communication in the presence of noise](http://fab.cba.mit.edu/classes/S62.12/docs/Shannon_noise.pdf)
    * Nyquist-Shannon 같은, 신호처리에 대한 중요한 개념들에 대해 설명한 논문 

***

### 이미지 신호처리란?
이미지 신호처리란, 입력 이미지를 픽셀 단위를 넘어서 일정 주파수를 가진 신호로 치환해서 해석하는 분야로, 대표적으로 Image Resizing / Motion Estimation / Image Enhancement / Compression 같은 분야에서 쓰입니다. 

얼라이어싱의 문제를 해결하려면, 이미지를 픽셀 단위에서만 생각하면 안 됩니다.
* 예를 들어 픽셀로만 생각하면, 1024 해상도의 이미지가 1픽셀 움직일때 얼라이어싱이 발생하는 상황이 발생했고, 이를 전 단계인 512 해상도의 이미지 레이어에서 수정하고자 하면, 해당 레이어는 **"0.5 픽셀"** 이동에 대한 문제를 해결해야 하는데, 픽셀 레벨에서는 이 문제를 절대 해석할 수 없습니다.
* 따라서 픽셀을 넘어서, **이미지를 주파수를 가진 신호처럼** 생각해야 한다고 저자들은 주장합니다.
    * (물론 그렇다고 입력이 픽셀 단위인건 아니고, **입출력은 어디까지나 픽셀**입니다. 다만 관점과, 그에 따른 네트워크의 변화를 고안할 때 이미지를 신호로 해석하는 것입니다.)

이미지를 신호처리적 관점으로 생각하려면, 선행조건으로 이산적(픽셀 차원)과 연속적(주파수, 위상 차원)간의 변환이 자유로워야 합니다.
* Nyquist-Shannon 공식 : 연속적 신호에서 샘플레이트 s 로 샘플된 이산 신호는, 0 ~ s/2 사이의 주파수를 가진 연속적 신호를 온전히 표현할 수 있습니다.
    *  최대 주파수 s/2 를 가진 연속적 신호를 표현하러면, 이산적 차원에서 s 의 샘플레이트로 샘플링 해야 합니다.
        * s의 샘플레이트로 샘플링 = (s,s) 해상도의 픽셀 이미지를 뜻합니다.  
    *  이산 --> 연속적 차원으로 복원시, 그 과정에서 대역폭이 s/2를 넘을 수 없습니다.
        * s/2를 넘으면 넘은 주파수 정보들은 얼라이어싱으로 됩니다.   
* 연속적에서 이산적으로, 그리고 그 반대의 변환이 되면, (이론적으로) 이제 픽셀 이미지가 아닌 연속적 신호상에 대해 연산을 할 수 있으며, 픽셀 이미지는 일종의 인코딩된 대체 표현으로 취급할 수 있습니다.

연속적 <--> 이산적 차원간의 변화를 다음과 같이 적어볼 수 있습니다.
* 샘플레이트 s : 연속적 신호의 샘플레이트이자, 이산적 신호의 크기
* 이산 신호 $Z[x]$ : 1/s 간격으로 분포되어 있는 디랙 델타 함수들의 2D 집합체
    * 디랙 델타 함수 : 적분이 일정한, 한 점을 연속적인 차원에서 표현하기 위한 분포로, **이산적인 데이터를 미분하기 위해서 사용합니다.**
	* (s, s) 해상도의 이미지가 있으면, 그 이미지는 1/s 간격으로 분포되어 있는 다양한 디렉 델타 함수들의 집합으로 표현할 수 있습니다.
	    * 이론적으로는 무한히 많은 디랙 델타 함수로 이미지를 표현하는 것이 가능합니다.
* 연속 신호 $z(x)$ : 이산적으로 표현되어 있는 이미지에서 우리들이 얻고 싶은, 주파수와 위상으로 이루어진 연속적인 신호

이산 신호에서 연속 신호를 계산 (복원?)하는건 다음의 과정을 통해 이루어집니다:
* Whittaker-Shannon 공식 : $z(x) = (\phi_s * Z)[x]$
* $\phi_s$ : 차단주파수 $s/2$를 가진 "이상적인 저역필터"로, 이산 신호 $Z[x]$에 convolve되어 연속적힌 신호를 얻으면서, 주파수 정보도 $s/2$ 미만으로 제한합니다.
    * $\phi_s(x) = sinc(sx_0) \cdot sinc(sx_1)$ $(sinc(x) = sin(\pi x) / (\pi x))$
    * 실제로는 후술할 요인들 때문에 이상적인 저역필터가 아닌, window function $w_K$ 가 곱해진 형태를 사용합니다.
* 이렇게 이산적 공간 안에서 계산된 연속 신호는 $[0,1]^2$ 크기의 unit square 안에 존재하게 되는데, convolution을 사용하기 때문에 $[0,1]^2$ 영역 밖의 이산 함수들도 연속 함수 계산에 영향을 미치게 됩니다. 
* 따라서 (s,s) 해상도의 (이산적) 픽셀 이미지로 같은 크기의 연속적 신호를 구하는건 충분하지 못하며, 이론적으로는 무한한 크기의 이산 함수 공간이 필요합니다.
* 저자들은 대신 일정 수준의 margin을 둬서 (코드 상에서는 4 방향으로 각각 10 픽셀) 이미지를 저장합니다. (예 : 256 --> 276 해상도)
		
이산 신호에서 연속 신호를 위와 같이 계산할 수 있으면, 연속 신호에서 이산 신호를 구하는 건 비교적 간단한 샘플링으로 이루어집니다:
* $Z[x] = (III_s \odot z)(z)$
* $III_s$ : 2D 디랙 델타 함수들의 집합으로, "픽셀의 중앙"에 위치하게끔 그 위치가 왼쪽 아래가 아닌, 중앙에 오도록 합니다.
    * $III_s = \Sigma_{X \in \mathbf{Z}^2}\delta(x - (X+\frac{1}{2})/s)$

이제 이산적 <--> 연속적 변환이 (이론상으로) 정보손실 없이 이루어질수 있다는걸 확인했으므로, 이제 픽셀 상의 이미지가 아닌 신호로서의 이미지에 대해 생각해볼수 있게 되었습니다.
* $z(x)$ : 실제로 우리가 마주하는 연속적 신호의 이미지
* $Z[x]$ : 원래의 연속적 신호를 해석하기 용이하도록 한 인코딩

***

### 딥러닝에서의 신호처리
아무리 앞서서 이론적으로 이미지를 신호의 차원으로 해석하는데 성공했어도, 딥러닝이 적용되는건 픽셀 단위의 이미지임은 변함이 없습니다.

그러나 이 딥러닝 연산도 신호처리적 관점으로 다시 써볼수 있습니다.
* 이산적 차원에서의 연산 : $Z'=\mathbf{F}(Z)$ ($Z, Z'$ : 이산적 feature map, $\mathbf{F}$ : 딥러닝 레이어 (conv, upsample, nonlinearity, etc))
* 연속적 차원에서의 연산 : $z'=\mathbf{f}(z)$ ($z, z'$ : 연속적 feature map, $\mathbf{f}$ : 연속적 차원에서의 딥러닝 레이어 (conv, upsample, nonlinearity, etc))
* 이제, 한 차원에서의 연산은 다른 차원에서의 연산으로 대체할 수 있습니다:
    * ($s$ : 입력 샘플레이트, $s'$ : 출력 샘플레이트)
    * $f(z) = \phi_{s'} * \mathbf{F}(III_s \odot z)$
    * $\mathbf{F}(Z) = III_{s'} \odot f(\phi_s * Z)$
    	* 이때, $f$는 얼라이어싱을 막기 위해, $s'/2$ 이상의 주파수 정보를 출력하면 안됩니다.

논문의 저자들은 이미지의 이동 / 회전시 발생하는 texture sticking 문제를 해결하기 위해 얼라이어싱을 잡는걸 목표로 했습니다. 이를 위해선, 다음의 조건들이 모든 레이어들에 걸쳐서 만족되어야 합니다:
* Equivariance : 딥러닝 연산 $f$와 이동/회전 (spatial transformation) $t$가 있고, 2D 공간 상에서 교환법칙 ($f \degree t = t \degree f$)을 만족할 경우, 해당 연산은 이동/회전에 대해 Equivariance를 지킵니다.
	* 이동에 대한 Equivariance (Translational Equivariance)
	* 회전에 대한 Equivariance (Rotational Equivariance)
	* 두 종류의 Equivariance에 대해 지켜야 할 법칙들이 다른 관계로, 저자들은 두개에 대해 별도의 configuration을 만듭니다.
* Bandlimit : 딥러닝 연산 $f$이 $s'/2$ 이상의 주파수 정보를 출력하지 않을시, 그 연산은 신호에서 얼라이어싱 없이 픽셀 이미지를 뽑아낼 수 있습니다.

그럼, 딥러닝 연산에 쓰이는 4가지 종류의 레이어들이 (Convolution, Upsampling, Downsampling, Nonlinearity) 등이 그 규칙을 따르는지, 따르지 않는다면 어떤 방식으로 고쳐야 하는지 알아보겠습니다.

#### Convolution
* $K$ : 이산적 커널, 입력 feature map과 동일한 그리드에 존재합니다 (샘플레이트 $s$).
* 이산적 차원 : $\mathbf{F}_{conv}(Z) = K * Z$
* 연산적 차원 : $f_{conv}(z) = \phi_s*(K*(III_s \odot z))$
	* convolution은 기본적으로 교환법칙을 지킵니다
	* $\phi_s * (III_s \odot z) = z$ (Identity operation)
	* 따라서, $f_{conv}(z) = K*z$ : 이산적 차원과 연속적 차원간의 연산에 차이는 없습니다.
* Translational Equivariance : (충족) 기본적으로 교환법칙을 지킵니다.
* Rotational Equivariance : (충족) 해당 
* Bandlimit : (충족) 입력 feature map 과 커널이 같은 그래드에 존재하기 때문에, 새로운 주파수 정보가 추가되지 않습니다.
	* Can be interpreted as sliding $K$ over $z$,  no new frequencies introduced --> Bandlimit requirement fullfilled.
* Naturally, convolution is commutative with translation, thus equivariant to translation.
* $K$ must be radially symmetric for rotation equivariance
	* Thus **1*1 convolution kernels** are viable choice, despite their simplicity.
#### Upsampling
* Upsampling only increases the output sampling rate $(s' > s)$, thus **it does not modifies $z$**. --> Translation and rotation equivariance fullfilled.
* continuous domain : $f_{up}(z) = z$
* discrete domain : $\mathbf{F}_{up}(Z) = III_{s'} \odot (\phi_s * Z)$ 
* If $s' = ns$, implement by **first interleaving $Z$ with zeros** to increase sampling rate, then **convolve with filter** $III_{s'} \odot \phi_s$
#### Downsampling
* Downsampling requires passing low pass filter to $z$ remove frequncies above new bandlimit $s'/2$, in order to represent signal in coarser $Z$.
* continuous domain : $f_{down}(z) = \psi_{s'}*z$
	* $\psi_{s} \doteqdot s^2 \cdot \phi_{s}$ : ideal low-pass filter, which is the normalized interpolation filter.
* discrete domain : $\mathbf{F}_{down}(Z) = III_{s'} \odot (\psi_{s'} * (\phi_s * Z))$
	* $\mathbf{F}_{down}(Z) = (1/s^2) \cdot III_{s'} \odot (\psi_{s'} * \psi_s * Z)$
	* $\mathbf{F}_{down}(Z) = (s'/s)^2 \cdot III_{s'} \odot (\phi_{s'} * Z)$
		* $\psi_{s} * \psi_{s'} = \psi_{min(s, s')}$
* Implement by first discrete convolution, then dropping sample points. --> Translation equivariance fullfilled.
	* For rotation, filter $\phi_{s'}$ must be **replaced with radially symmetric filter**, with disc-shaped frequency
	* Ideal filter : $\phi^{\degree}_{s} = jinc(s \|x\|) = 2J_1({\pi}s\|x\|)/({\pi}s\|x\|)$ (Bessel function first order)
#### Nonlinearity
* While nonlinearity don't commute with translation/rotation on $Z$, it does on $z$ (all pointwise function commute trivially with geometric transformations?) --> equivariant to translation and rotation
* However, **it may introduce arbitrarily high frequencies** that cannot be represented in the output (e.g. ReLU) 
* Solution : Eliminate such high frequencies with $\psi_s$
	* continuous domain : $f_{\sigma}(z) = \psi_s * \sigma(z) = s^2 * \phi_s * \sigma(z)$
	* discrete domain : $F_{\sigma}(Z) = s^2 * III_s \odot (\phi_{s} * \sigma(\phi_{s} * Z))$
		* Discrete operation cannot be done without entering $z$ once ; **approximate it.**
			* Upsample signal, Apply nonlinearity, Downsample. scale of 2 is sufficient)
		* Also use ideal filter discussed for downsampling for rotation
* Nonlinearity **is only operation capable of generating novel frequencies**
	* Limit range of such frequencies by **applying reconstruction filter w/ lower cutoff** than s/2 before final discretization
	* This lets us control level of new information for each layer

### Practical application to generator network
* The goal is to make all layers of G equivariant w.r.t the continuous signal; make continuous operation $g$ equivariant w.r.t rotations and translations.
	* $g(t[z_0]; w) = t[g(z_0; w)]$
* Metric : Peak Signal-to-noise ratio (PSNR) in decibel
	* $EQ_T = 10 \cdot \log_{10}(I_{max}^2 / \mathbb{E}_{w \sim \mathcal{W}, x \sim \mathcal{X}^2, p \sim \mathcal{V}, c \sim \mathcal{C}} \Big[(g(t_x[z_0]; w)_c (p) - t_x[g(z_0;w)]_c(p)])^2\Big])$
	* Each Images generated with $w \sim \mathcal{W}$, pixels sampled at mutually valid region $p \sim \mathcal{V}$.
	* $I_{max} = 2$ : dynamic range of generated images $-1 ... 1$
	* $t_x$ : spatial translation with offset $x \sim \mathcal{X}^2$
	* Rotation equivariance $EQ_R$ also analogous to $EQ_T$

#### Setting B : Fourier Features
* Replace learnt feature $z_0 (4*4*512)$ with Fourier features
	* Able to define spatially infinite map
	* Allows to compute $EQ_R$, $EQ_T$ without approximating $t_x$
* Frequency sampled uniformly, within frequency band $f_c = 2$ to match dimension $(4*4*512)$, keep frequencies fixed over training.
* Slightly improves FID, but not equivariant.

#### Setting C :  No Noise Inputs
* Random noise is irrelevant to our goal of equivariance.
* Slightly improves FID, but not equivariant.

#### Setting D :  Simplified Generator
* Remove features that deals with regularization and gradient control
	* Decrease mapping network depth
	* Remove mixing regularization, path length regularitzation
	* Eliminate output skip connections
* Introduce more direct gradient control
	* Calculate exponential moving average $\sigma^2 = \mathbb{E}[x^2]$ for all pixels in feature maps
	* Divide all feature maps by $\sqrt{\sigma^2}$
		* Bake this into weights
* Weakens FID, but helps equivariance

#### Setting E : Boundaries and Upsampling
* 

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgyNDU2MTE1NywtMTk5OTA3MjQ1MCwtMj
AxMDExMDMzOCwxODI3NDk2NjIsMTQ3MDcwOTI4LDEwNjMzMzc3
MzcsLTEwNjM1NTY5ODEsLTgxOTA5MTQzNyw3ODc5MjAxMTQsLT
Q1MTIyMDcxOCwtMTYyNjU0MjMxOCwtNzk4NjEwMzE3XX0=
-->
