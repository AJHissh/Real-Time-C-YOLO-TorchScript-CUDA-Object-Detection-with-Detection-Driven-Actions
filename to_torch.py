import torch
from ultralytics import YOLO

assert torch.cuda.is_available(), "CUDA not available!"

model = YOLO("C:/Users/user/model.pt").to('cuda')

model.export(
    format='torchscript',
    device=0, 
    imgsz=640,
    simplify=True,
    optimize=False,  
    workspace=4
)

ts_model = torch.jit.load('C:/Users/user/model.torchscript').to('cuda')

gpu_model_path = 'C:/Users/user/model.torchscript'
torch.jit.save(ts_model, gpu_model_path)

test_input = torch.rand(1, 3, 640, 640).to('cuda')
loaded_model = torch.jit.load(gpu_model_path).to('cuda')
output = loaded_model(test_input)
