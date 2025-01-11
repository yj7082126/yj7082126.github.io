---
layout: post
title: 'Birth and Death of a Rose'
date: 2025-01-10T09:10:00-04:00  
categories: blog  
tags: 
- 3D Generation
- 4D Generation
---

## Birth and Death of a Rose

| Name | Birth and Death of a Rose |
| --- | --- |
| Authors | Chen Geng, Yunzhi Zhang, Shangzhe Wu, Jiajun Wu |
| Institute | Stanford University |
| Conference | arxiv (2412.05278) |
| Links | [[ArXiV]](https://arxiv.org/abs/2412.05278) [[Project]](https://chen-geng.com/rose4d)  |
| Tl;dr | 4D Reconstruction 계열 논문. 장미가 피고 지는것처럼 사물의 모양이나 색깔등이 시간에 따라 변하는걸 Temporal Object Intrinsic 으로 정의, 이를 DINOv2 feature map 의 형태를 띈 Neural State Map 의 형태로 구현한다. 카메라 각도와 시간 입력을 받아 Neural State Map 을 출력하는 Neural Template 을 배우고, 이 모델을 바탕으로 SDS를 실행, 시간에 따라 변하는 3D 모델을 만든다 |

![image.png](/assets/img/20250110/image.png)

![image.png](/assets/img/20250110/image%201.png)

![image.png](/assets/img/20250110/image%202.png)

## 정리

| 무엇을 하고자 하는가 | 장미가 피고 지거나 양초가 녹는 것처럼, 시간에 따른 사물의 변화를 적용할 수 있는 3D (4D) 생성을 생성모델의 힘을 빌려 진행
새로운 정의 : 
- Temporal Object Intrinsic (TOI) : 시간에 따른 사물 본질의 변화 |
| --- | --- |
| 왜 이전에는 해결되지 않았나 | SDS는 사물의 relighting을 지원하지 않으며, 모션도 적용시킬수 없고, 무엇보다 Janus Problem 으로 보여지듯 3D 를 잘 모른다. 당연하지만 4D는 더 못할거다.
따라서 SDS를 사용하되, 카메라 각도와 시간에 대해 변하는 TOI 정보에 대해 gradient가 먹히도록 하는 새로운 구조가 필요하다.
새로운 정의 : 
- Neural State Map (NSM) : 위의 TOI 를 내포하여 TOI 를 3D 상으로 구현시킬수 있는 representation. 이 연구에선 DINO v2 Feature Map 을 바탕으로 함.
- Neural Template (NT) : 입력 카메라 각도와 시간에 대해 3D 모델에 대한 NSM 을 뱉어주는 모델 |
| 어떻게 해결하였나 | 1. 한 사물에 대한 Coarse Deformable Mesh (CDM) 를 생성한다.
- 디테일할 필요는 없고, 사물의 모양과 시간에 따라 바뀌는 특성이 반영된 수준이면 충분하다.
- CogVideoX를 통해 초기 비디오를 생성.
- 이 비디오의 canonical 프레임에 대해 Zero123, Imagedream 등을 사용해 초기 모델을 학습한다.
- 다른 시간의 프레임들에 대해 위의 모델을 변형시킬수 있는 Deformation Field 를 학습 (Optical Flow loss, ARAP 등을 사용)
2. 위의 CDM 에 대한 NT를 배운다
- 각 카메라 각도와 시간에 대해 CDM 을 렌더링한다.
- 3D Recon 을 통해 배운 모델들은 실제 이미지 분포 (DINOv2) 와 다를 가능성이 크다. 이를 해결하기 위해 LCM 모델을 사용해 렌더링된 이미지를 한번 고쳐준다.
- 이 이미지들에 대해 DINOv2 를 실행, 위의 NSM을 얻는다 |

## 느낀점

- 지금까지 비디오 생성 모델 기반으로 본것중에선 새로운 관점? (물론 이 연구자가 이쪽으로 많이 보긴 했다)


### Citation

```
@misc{geng2024birthdeathrose,
      title={Birth and Death of a Rose}, 
      author={Chen Geng and Yunzhi Zhang and Shangzhe Wu and Jiajun Wu},
      year={2024},
      eprint={2412.05278},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05278}, 
}
```