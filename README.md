# SOAP
Code for "SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition", accepted by ACM MM'24.
# How to Use?
Dataset Preparation -> Training -> Testing
## Dataset Preparation
### Dataset Downloading
[SthSthv2](https://markdown.com.cn) [Kinetics](https://markdown.com.cn) [UCF101](https://markdown.com.cn) [HMDB51](https://markdown.com.cn)
### Frame Decoding
You can download related code in this [release](https://github.com/wenbohuang1002/video_dateset_clip/releases/tag/data_splits). <br><br>
 The data splts .txt files 

  ```
  splits
   |_ hmdb_ARN
   | |_ test.txt
   | | |_ [class0]/[video0]
   | | |_ ...
   | | |_ [class1]/[video0]
   | | |_ ...
   | |_ train.txt
   | |_ val.txt
   |_ kinetics_CMN
   |_ ssv2_OTAM
   |_ ucf_ARN
  ```

The pre-processed dataset is organized with the following structure:

  ```
  datasets
   |_ Kinetics
   | |_ frames
   |   |_ [class_0}
   |   | |_ [video_0]
   |   | | |_ img_00001.jpg
   |   | | |_ img_00001.jpg
   |   | | |_ ...
   |   | |_ [video_1]
   |   | | |_ img_00001.jpg
   |   | | |_ img_00002.jpg
   |   | | |_ ...
   |   |_ [class_0}
   |   |_ ...
   |_ ...
  ```
# Training
# Testing
# Copyright
People from my old research group, if you open this repo. Thank and apologize to me (Wenbo Huang, code holder) by default. <br>
我以前课题组的人，倘若打开这一仓库。默认对我(黄文博，代码持有者)表示感谢并道歉。
