"""Archived, not maintained: train YOLOv8s from scratch on uno_dataset/ (uno_phase1.yaml).

Needs ultralytics and a CUDA GPU (device=0). See experiments/README.md.
"""

from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Small FROM SCRATCH (Kaggle Compliant)...")
    
    model = YOLO('yolov8s.yaml')

    print("Starting Phase 1 Training (Curriculum: Vocabulary)...")
    
    # Start the training loop
    results = model.train(
        data='uno_phase1.yaml',  
        epochs=300,              # Let it run wild
        patience=50,             # Stop if validation loss doesn't improve for 50 epochs
        imgsz=640,               
        batch=16,                
        device=0,                
        project='uno_project',   
        name='phase1_vocab',     
        save=True,               
        workers=6                
    )
    
    print("Phase 1 Training Complete! Weights saved in uno_project/phase1_vocab/weights/best.pt")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    main()

    
