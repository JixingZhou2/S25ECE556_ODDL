# Human Action Classification on Edge Devices

This repository documents the Human Action Classification Project, which aims to recognize and classify human actions such as stand, squat, and plank, and deploy the solution on an edge device, specifically a Raspberry Pi. The block diagram shows the entire flow.
<img width="262" alt="image" src="https://github.com/user-attachments/assets/eb9a4517-6e79-444b-926b-72c21318a3fe" />

# Model and training
We use YOLOv11-small as our base model, trained on the dataset from the Pose Counting Repetition Computer Vision Project. 

The training process is handled by src/train.py, which:
- Loads a pretrained YOLOv11s model.
- Fine-tunes it on the the human action dataset.
  <img width="670" alt="image" src="https://github.com/user-attachments/assets/eb7f715d-c594-4cf9-8176-c913437579fe" />

# Prune and quantization
Once training is complete, the model is pruned and quantized to optimize it for edge deployment. Since YOLO is a complex model in structure, we did not make many changes into the model layers, but focus more on layer pruning, more specificically convolutional layer pruning. 

This is done via the notebook src/yolo_prune_quantization.ipynb, which:
- Performs block-wise sensitivity analysis to help determine appropriate pruning ratios per block
  <img width="629" alt="image" src="https://github.com/user-attachments/assets/da57cd86-a42b-45ec-8510-f5b6d391e01b" />
  <img width="679" alt="image" src="https://github.com/user-attachments/assets/2bfc0bc0-01e0-4eeb-9b1a-3acf61540ed7" />
  It is shown that with deeper blocks in the model architecture, prune ratio needs to be smaller to keep more important information.
- Applies structured pruning to convolutional layers. Each block will be assigned a customized prune ratio depending on the sensitivity scan result.
- Conducts a global sparsity check after pruning.
- Quantizes the model from FP32 to FP16 during ONNX export.
  
# Deployment on Raspberry Pi
After pruning and quantization, the resulting ONNX model can be deployed on a Raspberry Pi. To run live inference from a camera feed, use:
python raspberry_pi/live_camera.py

This script performs real-time human action recognition directly on the Raspberry Pi using the optimized model. Below is the screenshot of model running on Raspberry Pi.
<img width="837" alt="image" src="https://github.com/user-attachments/assets/a71c449d-e40b-4de0-9f63-6acdd082a8f8" />

