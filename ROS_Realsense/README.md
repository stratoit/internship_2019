# ROS Systemt to integrate Intel D325 Depth Camera and Intel T265 Tracking Camera

```
$ roslaunch realsense2_camera rs_camera.launch camera:=d400 serial_no:=836612072015 filters:=pointcloud align_depth:=true
$ rosrun depthimage_to_laserscan depthimage_to_laserscan image:=/d400/depth/image_rect_raw camera_info:=/d400/depth/camera_info _output_frame_id:=d400_link _range_min:=0.16
$ rosrun hector_mapping hector_mapping scan:=/scan _base_frame:=d400_link _laser_min_dist:=0.16
$ roslaunch realsense2_camera rs_t265.launch camera:=t265 serial_no:=905312111935
$ rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 t265_pose_frame d400_link
```
