# Odometry and SLAM using ROS, Realsense D400 and Realsense T265

## Objective

After estimating dispairty using our own stereo camera we went on to analyze the very famous [Intel® RealSense™ D400 Depth Camera](https://software.intel.com/en-us/realsense/d400) and [Intel® RealSense™ T265 Tracking Camera](https://www.intelrealsense.com/tracking-camera-t265/) A replicate our models in the same way so that they can be tested against state of the art hardware and softwares. 

## Prerequisites and Installations
> Can only be done on Ubuntu 16.04 or 18.04 

Following are the steps to setup your system environment to replicate our Experiment

1. Install [librealsense](https://github.com/IntelRealSense/librealsense) from the follwing [link](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md). This SDK is required for viewing the images and extracting data from Intel® RealSense™ Cameras

2. Connect your camera and run the following command to see if the device is visible or not

    ```bash
    rs-enumerate-devices
    ```

3. Install [ROS](https://www.ros.org/) from the follwing links. 

    For Ubuntu 18.04 
    
    http://wiki.ros.org/melodic/Installation/Ubuntu
  
    For Ubuntu 16.04
    
    http://wiki.ros.org/kinetic/Installation/Ubuntu
    
    Follow all the steps prefectly and then setup your ROS Workspace according to the following
    
    http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment
    
4. Now Install Intel® RealSense™ ROS wrapper for running realsense on ROS

    Enter your ROS workspace
    ```bash
    cd ~/catkin_ws/src/
    ```
    Clone the latest Intel&reg; RealSense&trade; ROS from [here](https://github.com/intel-ros/realsense/releases) into 'catkin_ws/src/'
    ```bashrc
    git clone https://github.com/IntelRealSense/realsense-ros.git
    cd realsense-ros/
    git checkout `git tag | sort -V | grep -P "^\d+\.\d+\.\d+" | tail -1`
    cd ..
    ```
    Clone ddynamic_reconfigure into your workspace
    
    ```bashrc
    git clone https://github.com/pal-robotics/ddynamic_reconfigure.git
    ```
    Build the Package
    
    ```bashrc
    cd ..
    catkin_make clean
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
    catkin_make install
    ```
    With this you built your package
    
    The next steps need to be repeated everytime you attempt to run ROS Realsense package
    
    Source the user built packages into the terminal
    
    ```bash
    cd ~
    cd catkin_ws
    . devel/setup.bash
    ```
    Run Realsense&trade ROS package
    ```bash
    roslaunch realsense2_camera rs_camera.launch
    ```
    To Understand the rest of the package visit the [realsense-ros github page](https://github.com/IntelRealSense/realsense-ros) for everything
    
5. Install [depthimage_to_laserscan](http://wiki.ros.org/depthimage_to_laserscan) *(Optional)*

    ```bash
    cd ~/catkin_ws/src/
    git clone https://github.com/ros-perception/depthimage_to_laserscan.git
    cd ..
    catkin_make clean
    catkin_make
    catkin_make install
    ```
    
    This pakcage is s to convert the depthimage from Realsense&trade to a ROS understandable fake LaserScan [sensor_msgs/LaserScan](http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html)
    
6. Install [rtabmap-ros](http://wiki.ros.org/rtabmap_ros)

    For Ubuntu 18.04
    ```bash
    sudo apt install ros-melodic-rtabmap
    ```
    For Ubuntu 16.04
    ```bash
    sudo apt install ros-kinetic-rtabmap
    ```
## Running the Camera

The respective cameras can be run using the following command
```bash
roslaunch realsense2_camera rs_camera.launch serial_no:<serial no of camera>
```
Serial number of camera can be checked using
```bash
rs-enumerate-devices | grep -e 'Name\|Serial'
```
To run both the cameras simultaneously download the file names my_launch.launch and execute the following command
```bash
cd <directory where my_launch.launch is places>
. ~/catkin_ws/devel/setup.bash
roslaunch my_launch.launch serial_no_camera1:=<serial no. of t265 camera> serial_no_camera2:=<serial no. of d400 camera> use_rtabmapviz:=true use_scan:=false
```
Note: If you installed depthimage_to_laserscan then you can use ```use_scan:=true``` else set it to ```false```

## Experiments

1. #### Using Intel RealSense to plot the path and calculate distance travelled by the vehicle.
    - Zero displacement (theoretical) results using the application (10th Jul ‘19):
        - 8mm
        - 357mm
        - 480mm
    - 600mm displacement (theoretical) results using the application (8th Jul ‘19):
        - 604mm, with a 30mm error when traversing the vehicle in the reverse direction.
        - 606mm, with a 24mm error.
        - 580mm, with a 21mm error.
        - 584mm, with 10mm error
        - 577mm with zero error.
    - 1000mm displacement (theoretical) results using the application (8th Jul ‘19):
        - 857mm with 10mm error when traversing the vehicle in the reverse direction.
        - 925mm with a 40mm error.
        - 868mm with 23mm error.
        - 870mm with 16mm error.
        - 877mm with 6mm error.
        - 869mm with 23mm error.
        - 869mm with 21mm error.
        - 863mm with 14mm error
        - 863mm with 11mm error.
        - 833mm with 10mm error.
    - 1500mm displacement (theoretical) results using the application (8th Jul ‘19):
        - 1479mm with 17mm error when traversing the vehicle in the reverse direction.
        - 1475mm with 10mm error.
        - 1466mm with 15mm error.
        - 1506mm with a negative 10mm error.
        - 1467mm with 17mm error.
    - 1955mm displacement (theoretical) results using the application (10th Jul ‘19):
        - Curved path: 1938mm
        - Straight path: 1878mm
        - Looped path: 1855mm
    - 7440mm displacement (theoretical) results using the application (10th Jul ‘19):
        - Curved path: 7089mm
        - Straight path: 7158mm
2. #### Using Robot Operating System to plot the path and calculate the distance travelled by the vehicle
    - 1955mm total displacement (10th Jul ‘19):
        - Minimal curvature, direct displacement: 1990.13mm
        - Non-linear path: 1992.44mm
        - Non-linear looped path: 2032.97mm
    - 7440mm total displacement (10th Jul ‘19):
        - Minimal curvature, direct path: 7593.89mm
        - Non-linear path: 7668.79mm
    - Zero net displacement in a circular path (10th Jul ‘19):
        - ~0mm
        
|Sl No.|Theoretical Displacement (mm)|Displacement type|Predicted Displacement using ROS Viewer(mm)|Predicted Displacement using RealSense Viewer(mm)|Net Error in ROS Viewer (mm)|Net Error in RealSense Viewer(mm)|
|---|---|---|---|---|---|---|
|1.1|1995|Direct displacement|1990.13|1878|+4.87|+117|
|1.2|1995|Non-linear path|1992.44|1938|+2.56|+57|
|1.3|1995|Non-linear looped path|2032.97|1855|-37.97|+140|
|2.1|7440|Direct displacement|7593.89|7158|-153.89|+282|
|2.2|7440|Non-linear path|7668.79|7089|-228.79|+351|
|3.1|0|Circular |0|281.67 (avg.)|0|-281.67 (avg.)|

*Note:Error = Theoretical displacement - Predicted Displacement*

3. #### Simultaneously viewing the T265 camera output on ROS and RealSense viewer to check for discrepancies between the two
    
    **Experiment**: We moved the cart along the same path twice, once using ROS only and the other time using RealSense viewer.
    
    _We were unable to run the both of them simultaneously because while one of them runs, the other can not capture the input stream._

    **Result**: The ROS viewer and the RealSense viewer outputs were not consistent as mentioned in the table above.

4. #### Using the D400 camera with R-TabMap (visual odometry)
    
    **Experiment**: Map was built in ROS using point-cloud from D400 and odometry from D400.SLAM package used was R-TabMap and odometry was derived using visual odometry in the same package.

    **Results**: Map was very clean and improved over time as we iterated in the same environment multiple times.
    
5. #### Using the map from the D400 as the input for the T265 camera

    **Experiment**: Map was built in ROS using point-cloud from D400 and odometry from T265 in R-TabMap.
    
    **Results**: Map was not very clean compared to the map built using visual odometry from D400 without using the T265.
    
6. #### D400 for visual odometry on a **previously mapped environment**

    **Experiment**: Map was built in ROS using point-cloud from D400 and odometry from D400 and the same map was used later to check odometry in the same environment.
    
    **Results**: The odometry was perfect and better than T265 working alone or the visual odometry of the D400 camera in a new environment.
    
## Conclusion

-  D400 camera is the better choice for building maps of any environment over iterated loop closures (visual odometry performed using R-TabMap).
- Odometry results using the D400 camera were the most accurate results in a previously mapped environment. It also surpassed odometry results using the T265 camera in the same environment.
- Odometry results using the T265 camera with an input map from D400 gave the poorest results of all the combinations that we tester.
- Odometry error increases with an increase in the non-linearity of the paths taken.
- An unavoidable source of error is the human error which may influence readings within a -20mm to +20mm range. These include parallax effect, occluded view etc. 
- The errors are calculated at value when the cart speed is less than 400mm/s: results may not be replicated at higher speeds.

