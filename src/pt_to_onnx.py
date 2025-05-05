#!/usr/bin/env python3
# pttoonnx.py

import torch
from pathlib import Path
from ultralytics import YOLO

def export_to_onnx_manual(
    weights: str,
    onnx_path: str,
    imgsz: int = 640,
    opset: int = 12,
):

    device = torch.device('cpu')
    model = YOLO(weights)
    model.model.to(device).eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)

    torch.onnx.export(
        model.model,               
        dummy_input,               
        onnx_path,                 
        export_params=True,        
        opset_version=opset,       
        do_constant_folding=True,  
        input_names=['images'],    
        output_names=['output'],   
        dynamic_axes={             
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print(f'onnx file export to: {onnx_path}')

if __name__ == '__main__':
    weights_path = r'models\yolov11s-pos.pt'
    onnx_output = r'models\yolov11s-pos.onnx'
    export_to_onnx_manual(weights_path, onnx_output)
