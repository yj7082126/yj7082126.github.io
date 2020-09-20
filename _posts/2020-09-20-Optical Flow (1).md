---
title: “Optical Flow (1)”  
date: 2020-09-20T13:26:00-04:00  
categories:
-   blog  
tags:
-   machine learning
-   computer vision
-   optical flow
---

[업데이트 중]

### What is Optical Flow?

* Definition : Optical Flow is a vector map, describing the pixel motion between two consecutive frames.
	* For visualization, usually a HSV Color map is used, color for direction and saturation for magnitude.

### How to calculate it? Optical Flow Constraint
* Optical Flow Constraint: The value of two pixels, connected by a flow vector, should be equal.
	* $I (x, y, t)$ : Pixel value of pixel at location (x, y) of frame t
	* $( u, v)$ : 2 dimensional Flow Vector
	* $I (x, y, t) = I(x + u, y + v, t + 1) \\= I(x, y, t+1) + \frac{dI}{dx}u + \frac{dI}{dy}v$ (First-order Taylor Approximation)
	* $I_t + I_xu + I_yv = 0$
	* Limitation : due to the usage of the First-order Taylor Approximation, the following equation might be suitable for small displacements, but not for large.
* The Optical Flow Constraint is Underdetermined, and due to issues like Aperture Problem, additional constraints are needed to estimate the flow.

### Estimation : Lucas-Kanade
* $min_{u,v} \{ \Sigma_{(x', y') \in N^{2}} (I_t(x' , y') + I_x(x')u + I_y(y')v)^{2} \}$
	* $I^{1}_t + I^{1}_xu + I^{1}_yv = 0$
	* $...$
	* $I^{n}_t + I^{n}_xu + I^{n}_yv = 0$

### Estimation : Variational Method

* $\int_{\Omega} (|\nabla u|^2 + |\nabla v|^2) + (\lambda|I_t + I_xu + I_yv|^2) d\Omega$

* $min_{u,v,u',v'} \{ \int_{\Omega} (|\nabla u| + |\nabla v|) + (\frac{1}{2\theta}(u-u')^2 + \frac{1}{2\theta}(v-v')^2) + (\lambda|I_t + I_xu' + I_yv'|) d\Omega \}$
	* $min_{u',v'} \{ \int_{\Omega} (\frac{1}{2\theta}(u-u')^2 + \frac{1}{2\theta}(v-v')^2) + (\lambda|I_t + I_xu' + I_yv'|) d\Omega \}$ (w/ u & v fixed)
	* $min_{u,v} \{ \int_{\Omega} (|\nabla u| + |\nabla v|) +(\frac{1}{2\theta}(u-u')^2 + \frac{1}{2\theta}(v-v')^2) d\Omega \}$
 (w/ u' & v' fixed)
> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExOTAzOTgyMzBdfQ==
-->