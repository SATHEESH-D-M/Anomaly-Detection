# CountAI - Interview Assessment Task 

This README.md file will summarize all the files in this submission folder and will be a starting point to go through my solution.

### Report 
- refer to ```./report.pdf```

### Environment setup
- requires miniconda or anaconda package manager pre installed.
- use the ```./conda.yaml``` to create the conda environment

```
conda env create -f conda.yaml
```

- Then activate the environment 
```
conda activate  mlflow-env
```

### Accessing the model submitted.

- Start the mlflow server
```
mlflow server --host 0.0.0.0 --port 5001
```

- then, open ```http://localhost:5001/``` in the web browser.

- All of the experiments I tried and the model runs will be accessible form this UI.

- Final model (submission) is under ```artifacts/best_model``` in,
    - Experiment name : **post_baseline_ResNet**
    - Experiment id : **727670261461040353**
    - Run Name : **delicate-ray-801**
    - also registered the model as ```Resnet_final_submission```

- model input dimensions,
```python
input_size = (-1, 3, 720, 1270)
# torch format (batch_size, channels, height, width)
```

- input transformations to be used at the dataloader,

```python
from torchvision import transforms

input_transform = transforms.Compose([
        transforms.Resize((720, 1270)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
```
### Overview of the code files

There are 2 folder with code,
- ```./baseline``` has the code used for inital training of the models to achieve a baseline. Alongside the version of code with the parametes used for data augmentation. 

- ```./post_baseline``` has the code used for training the models after adopting corrections from the previous iteration. The submission model comes from here.

both has almost similar file organization. 

```bash
.
├── augmentation.py     # data_augmentation script
├── models                  
│   ├── ConvNeXt.py
│   ├── EfficientNet.py
│   ├── MobileNet.py
│   ├── ResNet.py
│   ├── ShuffleNet.py
│   ├── SqueezeNet.py
│   └── __init__.py
├── train_demo.ipynb        # training file
└── utils                   # custom helper functions used
    ├── __init__.py
    ├── datasets.py         
    ├── regularization.py   
    ├── seed.py
    ├── train_validate.py
    └── transfer_learning.py
```



