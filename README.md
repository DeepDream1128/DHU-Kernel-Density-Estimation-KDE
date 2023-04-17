**KDE REPORT**

211440128 赵伯远

210995119 吴文博

**  
**

Contents

[ABSTRACT](#abstract)

[1. INTRODUCTION](#introduction)

[2. THEORY OF KERNEL DENSITY ESTIMATION](#theory-of-kernel-density-estimation)

[2.1 Definition](#definition)

[2.2 Kernel Functions](#kernel-functions)

[2.2.1 Why we choose Epanechnikov Kernel?](#why-we-choose-epanechnikov-kernel)

[2.3 Gaussian Filter](#gaussian-filter)

[3. CODING](#coding)

[3.1 Three-dimension color KDE images](#_Toc130635232)

[3.2 Binary image detection results of the images](#binary-image-detection-results-of-the-images)

[3.3 Another kernel (Gaussian)](#another-kernel-gaussian)

[3.4 Another video](#another-video)

[4. RESULT](#result)

[4.1 Three-dimension colored KDE images (Epanechnikov Kernel) (Partial)](#three-dimension-colored-kde-images-epanechnikov-kernel-partial)

[4.2 Binary image detection results of the target (Epanechnikov Kernel) (Partial)](#binary-image-detection-results-of-the-target-epanechnikov-kernel-partial)

[4.3 Another kernel (Gaussian Kernel) (Partial)](#another-kernel-gaussian-kernel-partial)

[4.4 Another video (Partial)](#another-video-partial)

[4.4.1 Epanechnikov Kernel](#epanechnikov-kernel)

[4.4.2 Gaussian Kernel](#gaussian-kernel)

[5. ANALYSIS AND IMPROVEMENT](#analysis-and-improvement)

[5.1 Gaussian filter](#gaussian-filter-1)

[5.2 The impact of different calculation methods on computation time](#the-impact-of-different-calculation-methods-on-computation-time)

[5.3 Improvement](#improvement)

[ATTACHMENT](#attachment)

# ABSTRACT

Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability density function of a random variable. This report aims to demonstrate our result of ‘Using KDE to predict the movement of objects in the picture’. We also discuss the selection of kernel functions and bandwidth, as well as the limitations and advantages of using KDE.

# INTRODUCTION

Kernel Density Estimation (KDE) is a widely used technique in statistics and machine learning to approximate the probability density function (PDF) of a random variable. It is a non-parametric method, meaning it does not assume any specific distribution for the underlying data. KDE has gained popularity due to its simplicity, flexibility, and its ability to provide smooth and continuous estimates of the data distribution.

# THEORY OF KERNEL DENSITY ESTIMATION

## Definition

KDE can be defined as the sum of kernel functions centered at each data point. Mathematically, the kernel density estimate for a given point x is:

where:

is the number of data points

is thedata point

is the kernel function

is the bandwidth

When the bandwidth matrix is simplified to a single parameter, the familiar form is obtained:

The multidimensional kernel function can be represented by the product of kernel functions in each dimension:

## Kernel Functions

The kernel function K is a symmetric, non-negative function that integrates to one. Commonly used kernel functions include Gaussian, Epanechnikov, and Tophat kernels. The choice of kernel function can affect the smoothness and shape of the estimated density. In this project, we choose Epanechnikov and Gaussian kernel.

## Why we choose Epanechnikov Kernel?

1.  Optimal in the Mean Integrated Squared Error (MISE) sense: The EP kernel is optimal in minimizing the mean integrated squared error (MISE) for a given bandwidth when estimating the true underlying probability density function. This optimality means that, under certain conditions, it provides the lowest expected error among all possible kernel functions.
2.  Bounded support: Unlike the Gaussian kernel, the EP kernel has a well-defined edge, as its support is limited to the range [-1, 1]. This bounded support simplifies computations and makes the kernel function computationally efficient, especially for large datasets.
3.  Smoothness: The EP kernel is smooth, which leads to smooth and visually appealing density estimates. It is also differentiable, which can be important in certain optimization problems.
4.  Simplicity: The EP kernel has a simple mathematical form, making it easy to implement and understand.
5.  Faster convergence rate: The EP kernel has a faster convergence rate compared to some other kernel functions, such as the Gaussian kernel, which can lead to more accurate density estimates when using a finite sample size.

    The three-dimension Epanechnikov Kernel is: (whenis in the range of)

## Gaussian Filter

The Gaussian filter, also known as the Gaussian blur, is a type of image processing filter that is used to smooth or blur images. It is named after the Gaussian function, which is a mathematical function that closely approximates the normal distribution. This filter is widely used in computer vision, image processing, and signal processing for various purposes, such as noise reduction, edge detection, and feature extraction.

The Gaussian filter works by convolving the image with a Gaussian kernel, which is a matrix of values generated from the Gaussian function. The Gaussian function is defined as:

Here, (x, y) are the coordinates of a point in the image, and σ (sigma) is the standard deviation of the Gaussian distribution, which controls the amount of blur or smoothing applied to the image. The Gaussian kernel is generated by evaluating the Gaussian function for every point in a square grid, and then normalizing the kernel so that the sum of its elements is equal to 1.

When convolving the image with the Gaussian kernel, each pixel's value is replaced with a weighted sum of its neighboring pixels' values. The weights are determined by the Gaussian kernel, with pixels closer to the center of the kernel having higher weights than those farther away. This process effectively smooths the image by averaging the pixel values in a way that preserves the overall structure of the image, while reducing high-frequency noise and small-scale details.

**Example:**

![](media/2ebfdd17fca97fc262b8c2065917c221.png)

![](media/e0a37669b831b9eaccd2015e96207983.png)

Perform a Gaussian convolution on the pixel with a value of 226 at the position in the 4th row and 3rd column of the image. The calculation rule is to sum the pixel values in the neighborhood with different weights.

In actual calculations, for the pixel with a value of 226 at the position in the 4th row and 3rd column, Gaussian filtering is performed as follows:

Some advantages of using the Gaussian filter include:

-   Isotropic smoothing: The Gaussian filter smooths the image uniformly in all directions, which helps preserve the overall structure and shape of objects in the image.
-   Separable filter: The Gaussian filter can be separated into two one-dimensional filters, which can be applied sequentially to the rows and columns of the image. This reduces the computational complexity of the filter and makes it faster to apply.
-   Edge preservation: Although the Gaussian filter blurs the image, it does so in a way that preserves the overall structure and edges of objects, making it suitable for use in edge detection and feature extraction algorithms.
-   Tunable smoothing: The amount of smoothing applied by the Gaussian filter can be controlled by adjusting the standard deviation (σ) of the Gaussian function, allowing for customization of the filter's effect on the image.

# CODING

Platform: Visual Studio Code C++, MATLAB

Configuration: OpenCV 4.5.0, CMake

## Three-dimension color KDE images

1.  Import Images (train and test) and perform Gaussian filtering

    ![](media/03fa5f293209981f7847fb9868f9331e.png)

2.  Calculate the coefficients of the Epanechnikov kernel and then calculate Epanechnikov kernel

    ![](media/7a39be0a0c6746c79428314e900e98fa.png)

    ![](media/8d43cbc7aea6fbc4b57305bf1fb37dcf.png)

3.  Construct a KDE function using the Epanechnikov kernel

    ![](media/e845bcd8cb49b0c1cf06f24810554ca0.png)

4.  Calculate bandwidth h with Scott's Rule

    ![](media/af9dd5f4722472e8d5e353e5f87d8bf3.png)

5.  Perform KDE function
6.  Output the density value of each test pixel to a file

    ![](media/0ae1011b5a7f335389c6e6081c54039c.png)

7.  Import the file to MATLAB and generate a Three-dimension colored KDE image

    ![](media/2f6d1d0469c4a5e435dd3e7808b14d6a.png)

## Binary image detection results of the images

1.  Import Images (train and test) and perform Gaussian filtering
2.  Calculate the coefficients of the Epanechnikov kernel and then calculate Epanechnikov kernel
3.  Construct a KDE function using the Epanechnikov kernel
4.  Calculate bandwidth h with Scott's Rule
5.  Perform KDE function
6.  Set a threshold, if the value of the density is larger than it, set the color of the pixel to black (background) or white (foreground)
7.  Generate the images and save them

## Another kernel (Gaussian)

Construct another KDE function using the Gaussian Kernel

![](media/7ad859cd797032a30f42de8914eab5c6.png)![](media/eb13dea47df29b9846801f0021bd1258.png)

## Another video

Source: recorded by ourselves

Resolution: 3840\*2160 (4K), 25.00 frames/second, Color Depth: 8bit

Frame sampling:

Platform: Python 3.9

Configuration: OpenCV 4.7.0

Code:

![](media/06ba1847fa289c3cf55b79cc5ae54880.png)

# RESULT

## Three-dimension colored KDE images (Epanechnikov Kernel) (Partial)

![](media/d9ed1f59bc1a803d9b2c39e321023515.png) ![](media/4661624b33b0c2aeb8ace5331d959445.png)

(without Gaussian filter) (with Gaussian filter)

## Binary image detection results of the target (Epanechnikov Kernel) (Partial)

![](media/5219c5685402faaa04e04a82e45927a3.jpeg) ![](media/2d8bf82ce7b149706dd336101eca4d93.jpeg)

(without Gaussian filter) (with Gaussian filter)

## Another kernel (Gaussian Kernel) (Partial)

![](media/c4f2e70bd80df4f40c0c484c9840d6ba.png) ![](media/a7c789d326fe3cddfa150ab0fee3b233.jpeg)

(without Gaussian filter)

![](media/f2994b77e72010aaac4721531e095349.png) ![](media/dbdcbbdd487b8734f1f77dcda0a4519b.jpeg)

(with Gaussian fliter)

## Another video (Partial)

## Epanechnikov Kernel

![](media/178251cab3381d8e673be3d2daf8d950.png)  **![](media/712be14e5c1abc015c7168cc42d43181.jpeg)**

(without Gaussian filter)

![](media/42552e512b9e0954eef76a3af2d5f1ea.png) **![](media/d8043682709cc4535a6943496255c7d6.jpeg)**

(with Gaussian fliter)

## Gaussian Kernel

![](media/7c1a3600e9ebdcca34ee131fd0d81029.png) ![](media/963a29062e31d3449c9831cf90036833.jpeg)

(without Gaussian filter)

![](media/f27e8655fef49bbcf700c6c56dae0ba8.png) ![](media/ddf6fc032472e493cf0912baf504c07d.jpeg)

(with Gaussian filter)

# ANALYSIS AND IMPROVEMENT

## Gaussian filter

According to the results shown in sections 4.1, 4.2, and 4.4, we find that applying Gaussian filtering to low-resolution images can yield more accurate results(4.1, 4.2). The main reason is that Gaussian filtering can eliminate the influence of external factors such as camera movement and wind, which are beyond our control. In high-resolution images, however, the blurring effect of Gaussian filtering between pixels is not significant, so its impact on the results is minimal. (4.4)

## The impact of different calculation methods on computation time

We have exported the program runtime [log.txt](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/2022/kde_last/log2.txt) and found that at the same resolution and bandwidth, the Gaussian kernel requires a longer computation time than the Epanechnikov kernel. At the same time, the higher the bandwidth, the less time is required. At the same resolution, for images with Gaussian filtering applied, Scott's Rule calculates a smaller bandwidth. When setting the bandwidth manually (i.e., not using Scott's Rule to calculate the bandwidth), the program's runtime is significantly reduced. Therefore, it can be concluded that the calculation of bandwidth consumes most of the computation time.

![](media/51d9f03ebe8eaa37a6d1fda69fc51068.png) ![](media/914f00c0a962173ca8548419433dc24e.png)

(log with Scott’s Rule)

![](media/560ae84937f01286d883a72b2fe8b544.png) ![](media/11ec5f8b3fe2f5be7145aa5f1b315fc7.png)

(log without Scott’s Rule)

## Improvement

We found that if the frame number difference between the training images and the testing images is too large (with a larger gap in the frame count), subtle camera movements can cause pixel-level motion, leading to errors in the calculated results. We plan to investigate an algorithm in the future to eliminate the errors caused by such subtle motion to obtain more accurate results.

# ATTACHMENT

Code:

The latest Code: [kde_last\\KDE\\kde_last.cpp](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/KDE/kde_last.cpp)

Former Code: [kde_last\\KDE\\kde_v1.0.cpp](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/KDE/kde_v1.0.cpp) to [kde_last\\KDE\\kde_v2.9.cpp](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/KDE/kde_v2.9.cpp)

.mp4 to .jpg: [ep trans.py](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/ep%20trans.py)

Plot (MATLAB): [kde_last\\RESULT\\KDE.m](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/RESULT/KDE.m)

CMakeLists: [kde_last\\KDE\\CMakeLists.txt](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/KDE/CMakeLists.txt)

Results:

Data and Images: [kde_last\\RESULT](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/RESULT)

Time logs: [kde_last\\log2.txt](../Documents/WeChat%20Files/wxid_73hkttxy5hn721/FileStorage/File/2023-04/kde_last/log2.txt)
