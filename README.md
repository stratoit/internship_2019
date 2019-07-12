# Robotic Odometry

The following repository is controlled by **[Strato-IT](https://stratoitkorea.com)** and was created under the
the direction of **Mr. Younghyo Kim, CEO, Strato-IT** (yhkim@strato-it.co.kr) during the period of May to July 2019. This repository was created by interns in order to explore the practical applications of Visual and Interial Odometry in known and unknown environments using state of the art techniques. 

## Objective

To achieve stereo visual odometry using a pir of stereo camera and test different existing alogrithms and hardware in the field of Robotic Odometry

## Different ways of odometry 

### Odometry using Inertial Sensors

This is still considered to be the most accurate measurement of Odometry using IMUs which give accurate velocity, acceleration and gyrosopic pose. This pose estimation works best as its based on direct measurement of motion parameters and estimating position using that.

### Visual Odometry

This method uses images to measure the motion of the camera. It matches pixel positions with respect to each other to estimate the rotational and transaltional transform of one image to other and that inherently is also the translation and rotation of the camera. It is of two types

- #### Monocular

    Uses single camera to estimate position of camera using image at two different times. Lacks undersatnding of Scale and one image cannot be used to determine depth of objects being used as refrence for estimating transformations

- #### Stereo Pair

    Uses two camera which focus on same scene. Two cameras who are placed in a known way attached to a stable frame. help us determine the depth of objects being placed in the environment using pixel shift and disparity estimation. This set doesn't suffer from scale invariance and gives transforms that are actual distances in the real world. For this method to work the position of cameras w.r.t to each other shouldn't change.

### LIDAR and PointCloud Odometry

Cameras like IR Camera or 3D Lidar can also be used to estimate Odometry. Here we already have the depth information so alognment of points in 3d space is easier because of the fourth depth channel which provides a lot of data about placement of objects in environment directly without any pre processing. With today's technology this method performs at par with Inertial Odometry because the robust data being achieved.

## Project

Our project was divided into three parts 
### Research

1. Study famous state of the art papers both in mathematics and machine learning to estimate odometry using stereo image pairs.
2. Found some really good papers with good results having top rankings on KITTI Dataset. 
3. The papers are explained in the [Research Papers](https://github.com/stratoit/internship_2019/tree/master/Realsense) folder

### Implementing Machine Learning models for disparity estimation of stereo pairs

1. Disparity estimation was the first step in assessing odometry using stereo pairs as it gave an idea of depth of the images being captured.
2. We implemented two state of the art machine learning papers in the field of disparity estimation.
3. All the information regarding the implementations of disparity estimation papers are in the [Disparity Map Estimation](https://github.com/stratoit/internship_2019/tree/master/Disparity_Map_Estimation) folder. 
4. After dispairty estimation we can pass it to famous visual odomerty algorithms like rtabmap or orb-slam to estimate odometry

### Using Already present hardware and software and study thier accuracy

1. Tested the acuracy of Intel&reg; Realsense&trade; T265 Tracking camera using ROS and Realsense viewer
2. Tested the accuray in mapping and odometry using Intel&reg; Realsense&trade; T265 Tracking Camera and D435 Depth Camera in different scenarios and combinations using ROS.
3. All information pertaining to this can be found in [Realsense](https://github.com/stratoit/internship_2019/tree/master/Realsense) folder.

## Future Prospects

1. The implemented Dispairty map estimators are only python codes without training so they need to be trained first.
2. The trained models can be packged in a ROS Wrapper can be directly fed to Rtabmap or ORB-SLAM to test its accuracy compared to other hardware and software techniques available.
3. Wheel odometry of the bot can also be used to coorect the estimation of visula odometry which will be of great help. 
4. As Realsense&trade; fuctions really well in known visual environment its data can be used to refine the odometry from T265 to achieve more accurate odometry.

-------

## Contributors

**[Apoorva Kumar](https://cybr17crwlr.github.io)** : Pre-Final Year Undergrad, B.Tech in Electronics and Electrical Engineering, IIT Guwahati

**[Aniket Mandle](https://linkedin.com/in/aniketmandle)** : Pre-Final Year Undergrad, B.Tech in Mechnical Engineering, IIT Guwahati

**[Abhishek Tiwari](https://linkedin.com/in/abhishektiwari18448)**: Pre-Final Year Undergrad, B.Tech in Mechanical Engineering, IIT Guwahati

**[Avish Kabra](https://linkedin.com/in/avish-kabra)**: Pre-Final Year Undergrad, B.Tech in Electroncis and Electrical Engineering, IIT Guwahati

-------

## License

This project must not be replicated without necessary permissions from Strato-IT or the authors.
