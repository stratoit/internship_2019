# Prerequisites of Odometry

1. Rectification 
    * Each pixel's match in another image can only be found on a line called the epipolar line.
    * Even with high-precision equipment, image rectification is usually performed because it may be impractical to maintain perfect coplanarity between cameras.
2. Edge Preserving Filtering
    * Laplacian of Gaussian filter ( LOG ) 
    * Difference of Boxes Filters: Shadow Suppression and Efficient Character Segmentation
3. Correlation: Constructing Disparity
    * We do not need to search the entire right image for a particular feature in the left image, We only need to search along the epipolar line.

# Papers and releated research 

Here we collect all the info on the papers on the course of achieving this project.

## Content

- [Stereo Matching with Color and Monochrome Cameras in Low-light Conditions](#stereo-matching-with-color-and-monochrome-cameras-in-low-light-conditions)
- [DSVO (Direct Stereo Visual Odometry)](#dsvo-direct-stereo-visual-odometry)
- [Open StereoNet](#open-stereonet)
- [Real-Time Stereo Visual Odometry for Autonomous Ground Vehicles](#real-time-stereo-visual-odometry-for-autonomous-ground-vehicles)
- [iResNet](#iresnet)

## Personal Views on the paper and its use

### [Stereo Matching with Color and Monochrome Cameras in Low-light Conditions](https://sunghoonim.github.io/assets/paper/CVPR16_RGBW.pdf)

Multi-modal and multi-spectral imaging approaches such as a color and infrared camera pair and cross-channel matching have been proposed. However, these approaches require high manufacturing cost and specialized hardware.
Here they exploit the fundamental trade-off between color sensing capability and light efficiency of color cameras and monochrome cameras, respectively. Because monochrome cameras respond to all colors of light, they have much better light efficiency than Bayer-filtered color cameras.

![Stereo Matching with Color and Monochrome Cameras in Low-light Conditions](images/Monochrome.png)

#### Approach
1. First decolorize the color input image because two cameras have different spectral sensitivities and viewpoints.
2. Then, estimate disparities based on brightness constancy and edge similarity constraints,
3. Retain reliable correspondences with a left-right consistency check and aggregate them from all candidate decolorized images.
4. After that, this set of reliable correspondences is used to augment additional correspondences by iterative gain compensation and disparity estimation.
5. To achieve robust stereo matching results, we combine two complementary costs; the sum of absolute differences (SAD) as a brightness constancy measure and the sum of informative edges (SIE) as an edge similarity measure.
6. Given the grayscale input and aligned decolorized image, we match the brightness of the input image to the decolorized image by estimating a local gain map. Because our decolorization is performed to preserve the contrast distinctiveness of a color image, it can capture important local edges better than the grayscale input image, where edges may be ambiguous due to the mixing of spectral information. Therefore, this iterative process provides increases in the number of reliable correspondences.

-------
### [DSVO (Direct Stereo Visual Odometry)](https://arxiv.org/pdf/1810.03963.pdf)

-------
### [Open StereoNet](https://arxiv.org/pdf/1808.03959v1.pdf)

-------
### [Real-Time Stereo Visual Odometry for Autonomous Ground Vehicles](https://www-robotics.jpl.nasa.gov/publications/Andrew_Howard/howard_iros08_visodom.pdf)

-------
### [iResNet](https://arxiv.org/pdf/1712.01039.pdf)
