# Unitree GO2 Deploy

This repo is a Unitree GO2 Deployment which have been trained by IsaacGym.

We are supporting deployment in a Mujoco simulation.

### 1. Installation

Before installing this repo, you must install a [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco).

`git clone https://github.com/LPatstar/go2_deploy_gym.git`

`cd go2_deploy && pip3 install -e .`


### 2. How to use

### 2.1. Create Terrain

直接用core/go2里的terrain_generator.py即可

### 2.2. Deploy mujoco

`python go2_deploy.py --load_model [Your Path]/model_xxxx.pt --use_camera`


### Acknowledgement

Thanks to their previous projects.

1. @machines-in-motion [repo](https://github.com/machines-in-motion/Go2Py)

2. @NVlabs [repo](https://github.com/NVlabs/HOVER)

3. @eureka-research [repo](https://github.com/eureka-research/eurekaverse)

4. @boston-dynamics [repo](https://github.com/boston-dynamics/spot-rl-example)

5. @itt-DLSLab [repo](https://github.com/iit-DLSLab/gym-quadruped)

```
6. Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/go2_parkour_deploy
```
