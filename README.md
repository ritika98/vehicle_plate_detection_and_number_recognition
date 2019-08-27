# Vehicle plate detection using YOLOv3 and number recognition using DeepTEXT


## Setup and Usage

Step 1:
__Clone/Download the repository:__
git clone https://github.com/ritika98/vehicle_plate_detection_and_number_recognition.git

Step 2:
__Download weights of YOLOv3 for license plate detection and DeepText weights and put them inside the config folder:__
https://drive.google.com/open?id=1kIxP8C_b978f3SRDgDiMwo99oH_zpTBK

Step 3:
__Go inside the downloaded repository:__
		cd vehicle_plate_detection_and_number_recognition/

Step 2: 
__Set up virtual environment using conda:__
		conda env create --name your_env_name --file conda_environment.yml
         
Step 4: 
__Activation of the environment:__
		conda activate your_env_name
           
Step 5: 
__To use YOLOv3 detector script:__
python yolov3_detector.py  --image media/test.jpg --yolo config/

