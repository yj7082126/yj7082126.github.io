
title: “Optical Flow (1)”  
date: 2020-09-20T13:26:00-04:00  
categories:

-   blog  
    tags:
-   machine learning
-   computer vision
-   optical flow

----------

[업데이트 중]

### What is Optical Flow?

* Definition : Optical Flow is a vector map, describing the pixel motion between two consecutive frames.
	* For visualization, usually a HSV Color map is used, color for direction and saturation for magnitude.

### How to calculate it?
* Optical Flow Constraint: The value of two pixels, connected by a flow vector, should be equal.
	* $I (x, y, t)$ : Pixel value of pixel at location (x, y) of frame t
	* $( u, v)$ : 2 dimensional Flow Vector
	* $I (x, y, t) = I(x + u, y + v, t + 1) \\= I(x, y, t+1) + \frac{dI}{dx}u + \frac{dI}{dy}v$ (First-order Taylor Approximation)
	* $I_t + I_xu + I_yv = 0$
	* First-order Taylor Approximation 을 사용했기 때문에, 작은 움직임 (short displacement) 에는 적합할지 몰라도, 큰 움직임 (large displacement) 에는 적합하지 못하다.

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU3NTM1ODk2NV19
-->