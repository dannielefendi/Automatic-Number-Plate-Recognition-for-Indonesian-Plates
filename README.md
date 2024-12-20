# Automatic-Number-Plate-Recognition-for-Indonesian-Plates (ANPR)
 License Plate Recognition for Indonesian Plate with YOLOV11 Model and PaddleOCR and save it to CSV File

# Example
 The plate was detected by the model <br /> <br />
![image_with_boxes](https://github.com/user-attachments/assets/6d0f3f10-4b5d-4fdb-abee-6edd2b6e7336)

# Model
The model was trained with Yolov11 using this [car plate dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) 

# Project Setup
 - Make an environment with python=3.10 using the following command <br />
   ```
   python3 -m venv virtualenvname
   ```
  
 - Activate the environment <br />
   ```
   source /path/to/venv/bin/activate
   ```
  
- Install the requirement library for the Environment <br />
   ```
   pip install -r requirement.txt
   ```
  
- Run the anpr.py with image sample in the zip file <br />
  ```
  python anpr.py
  ```
