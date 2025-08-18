## This is a program that detects objects defined in a yolo model based on one or more classes/properties. 

This script is specifically tailored to detect one class.

Here is an example command to train a yolov8n model 
```
yolo task=detect mode=train model=yolov8n.pt data="C:/Users/user/model/data.yaml" epochs=30 imgsz=640 device=0 lr0=0.001 workers=8 cos_lr=True optimizer=AdamW mixup=0.1 lrf=0.01 perspective=0.0005 weight_decay=0.001 hsv_v=0.4 hsv_s=0.7 degrees=0.5 translate=0.1 hsv_h=0.015 scale=0.5 shear=0.1 mixup=0.1 mosaic=1.0 amp=True
```

This will then create a .pt file which you can turn into a torchscript file via to_torch.py.

### To build the script make sure you have the following installed:
```
Visual Studio 17 2022 + Windows SDK v10 
Libtorch 
Cuda 
Cmake 
OpenCV 
NVTX_v3
```

1. Place all files (CMakeLists.txt, main.cpp, dxgi .....) inside your working directory

2. Create a build directory and cd into it then run the following:

```
cmake .. -G "Visual Studio 17 2022" -A x64
```

3. Then run the following to build the program
```
cmake --build . --config Release --parallel 8
```
To rebuild after making changes, you can:
```
cmake --build . --target clean
```
then build again with:
  ```
  cmake --build . --config Release --parallel 8
```

4. The .exe file will be placed in the Release folder

Additional Steps:

1. Place the torchscript file within the Release folder where the .exe file was created
2. Move required files from Libtorch, Cuda, OpenCV, NVTX and Visual Studio folders into the Release folder
