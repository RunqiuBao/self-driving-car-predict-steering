# self-driving-car-predict-steering
Predicting steering angle for self driving car using deep learning

## pre-requisites:
1. install ros
2. install numpy                   
3. install matplotlib                    
4. install pandas                     
5. put the bag file in the dataset folder with name "dataset.bag"                   

### visualise the camera input               
run `roslaunch dataprocess visualise_camera_images.launch`               
this should launch rviz displaying the camera data.


### getting the data out of bag file
**for generating images from cameras and csv files**                       python
run `python bag_extract_data.py`                            
this should extract all the images into corresponding folders *left*, *right*, *center* and the folder *yaml_files* will contain the required csv files, the **final_interpolated.csv** will contain all image file names with corresponding data (steering angle, acceleration etc.) values.                           

**for only generating image files**                        
run `python bag_extract_image_data.py`                               
this will extract all the images into corresponding folders *left*, *right*, *center* and genrate a text file                        *image_steering_angle.txt* with values   


### training 
**for training on center camera images**             
run `python train_center.py`   
*change the epochs and batch size accordingly*

**for training using all the three camera images**                
run `python train_merged.py`          
*change the epochs and batch size accordingly*

### testing
run `inference.py`
this should load the weights and predict steeering angle on the test data


#### Notes:
1. The .yaml, .png, .bag files are ignored in git

runqiu:/left_camera/image_color
# 图像输入需要正则化
# 参数需要clip，loss过大

