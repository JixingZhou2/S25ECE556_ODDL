from ultralytics import YOLO

def run_training():
    from ultralytics import YOLO

    # Load a model
    model = YOLO("E:/AI/Yolo/models/yolo11s-pose.pt")
    #model.val(data="G:/AI/Dataset/Pose Counting Repetition.v2i.yolov8/data.yaml", conf=0.25, iou=0.65, device="0")
    
    model.train(
        data="G:/AI/Dataset/Pose Counting Repetition.v2i.yolov8/data.yaml",
        epochs=50,             # 
        batch=8,                # 
        imgsz=640,              # 
        device="cuda",          
        workers=4,              # 
        optimizer="SGD",        
        lr0=0.0005,               # 
        momentum=0.937,         # 
        weight_decay=0.0005,    # 
        cos_lr=True,            # 
        lrf=0.001,               # 
        patience=50,            # 
        mosaic=1.0,             # 
        hsv_h=0.015,            # 
        hsv_s=0.7,              # 
        hsv_v=0.4               # 
    )
    #results = model.train(data="coco8.yaml", lr0=0.001)
    

if __name__ == '__main__':
    run_training()