# YOLOv2 with TensorFlow 2.0 and Keras

This repo can be used for training a YOLOv2 based object detection model. The repo uses TensorFlow 2.0 and Keras for training and implmentation. This repo does not belong to a specific project. So it would be interesting what all projects do people come up with for solving computer vision problems.

For using this repo, the following list of commands could be used

Step 1: Initiate a virtual environment on your local machine
~~~
virtualenv YOLOv2_TensorFlow_VENV
cd ./YOLOv2_TensorFlow_VENV
~~~

Step 2: Clone this repository
~~~
git clone https://github.com/chatterjeesandipan/YOLOv2_TensorFlow.git
~~~

Step 3: Install the necessary packages mentioned in the file "requirements.txt"
~~~
pip install -r requirements.txt
~~~

Step 4: Clone the OIDv4 Toolkit for obtaining training data
~~~
git clone https://github.com/EscVM/OIDv4_ToolKit.git
~~~

Step 5: Download training data for a sample set of classes, say Person Woman Car Tree Truck
~~~
python ./OIDv4_ToolKit/main.py downloader --classes Person Woman Car Truck Tree --limit 3000 --type_csv all --multiclass 1 -y
~~~

Step 6: Run the training code
~~~
python YOLOv2_MasterCode.py
~~~

Step 7: Tensorboard visualization: Open a new terminal and activate the virtual environment created above. Navigate to the repo directory. Navigate to the folder ./LOGS/{Project name set by you}/Train_{datetime_stamp}. Run the following command
~~~
tensorboard --logdir=. --host localhost --port 8088
~~~

