# Disparity Estimation

We tried 3 methods to estimate dispairty from stereo images

1. #### OpenCV Method
2. #### iResNet
3. #### PSMNet

## Explanation

### OpenCV Method

This method uses the inbuilt OpenCV Methods to estimate Disparity. It uses two methods namely
- Block Matching
- Semi Block Matching

#### Result

The output was very good on Middlebury dataset but failed miserably on real life images 

### iResNet 

This method was developed on a old algorithm called DES-Net *(Implementation for both included in the folder)*. It uses refinment and upconvolution with skip connections to bring high level data from old layers and refine the output. The best part of this alogorithm is it does everything from feature detection to dispairity estimation in a single network rather than using different networks for each of them. But this networks need really high training time and so we were not able to test it on any dataset. 

The [iResNet](iResNet) folder contains network for a stereo camera pair having input isze of *1280x720* and a complete code ito train the data on KITTI Dataset for testing.

#### Result

Cannot test completely because of high training time. Expected to perform well because initial traning on small set gave good results.

### PSMNet

This method was also listed to have results almost equivalent to iResNet and also had pretained weights for testing on KITTI Dataset.All the working files are stored in the folder [PSMNet](PSMNet).  

#### Result

The output on KITTI Dataset was staisfactory and can be used to replace mathematical method. But the FPS will be very less because the network is computationally heavy

Out noted time taken for one forward pass was
- CPU : 24 sec
- Nvidia GTX 970M 6GB : 1.5 sec
- Nvidia RTX 2080Ti 11GB : 0.4 sec
