# Visual Skill Chaining for Long-Horizon Robot Manipulation

## Installation

0. Clone this repository and submodules.
```bash
$ git clone --recursive git@github.com:clvrai/skill-chaining.git
```

1. Install mujoco 2.1 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
Note that the code is compatible with **MuJoCo 2.0**, which supports Unity rendering.
```bash
# download mujoco 2.1
$ mkdir ~/.mujoco
$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco_linux.tar.gz
$ tar -xvzf mujoco_linux.tar.gz -C ~/.mujoco/
$ rm mujoco_linux.tar.gz

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

2. Install python dependencies
```bash
$ sudo apt-get install cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev

# software rendering
$ sudo apt-get install libgl1-mesa-glx libosmesa6 patchelf

# window rendering
$ sudo apt-get install libglfw3 libglew2.0
```

3. Install furniture submodule
```bash
$ cd furniture
$ pip install -e .
$ cd ../method
$ pip install -e .
$ pip install torch torchvision
```

## State-Based Skill Chaining
```
# Generate demonstrations for sub-task one
python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --start_count 0 --phase_ob True

# Training the first policy
mpirun -np 16 python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 --num_connects 1 --run_prefix p0 --gpu 0 --wandb True --wandb_entity arthur801031 --wandb_project skill-chaining

# Collect successful terminal states from the first sub-task policy
python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 --num_connects 1 --run_prefix p0 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt

# Generate demonstrations for sub-task two
python -m furniture.env.furniture_sawyer_gen --furniture_name chair_ingolf_0650 --demo_dir demos/chair_ingolf/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0 --start_count 1000 --phase_ob True

# Training the second policy
mpirun -np 8 python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 1 --preassembled 0 --run_prefix p1 --load_init_states log/chair_ingolf_0650.gail.p0.123/success_00011468800.pkl --gpu 0 --wandb True --wandb_entity arthur801031 --wandb_project skill-chaining

# Collect successful terminal states from the second sub-task policy
python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 1 --preassembled 0 --run_prefix p1 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt

# Training the state-based skill-chained policies
mpirun -np 8 python -m run --algo ps --furniture_name chair_ingolf_0650 --num_connects 2 --run_prefix fix2 \
--ps_ckpts log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt,log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt \
--ps_load_init_states log/chair_ingolf_0650.gail.p0.123/success_00011468800.pkl,log/chair_ingolf_0650.gail.p1.123/success_00017203200.pkl \
--ps_demo_paths demos/chair_ingolf/Sawyer_chair_ingolf_0650_0,demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --gpu 0 --wandb True --wandb_entity arthur801031 --wandb_project skill-chaining

# Evaluate state-based policies
python -m run --algo ps --furniture_name chair_ingolf_0650 --ps_demo_paths demos/chair_ingolf/Sawyer_chair_ingolf_0650_0,demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 2 --run_prefix fix2 --is_train False --num_eval 100 --record_video False --ps_ckpts log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt,log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt --init_ckpt_path log/chair_ingolf_0650.ps.fix2.123/ckpt_00002048000.pt --seed=1234

# Generate expert trajectories for sub-policy one
python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 --num_connects 1 --run_prefix p0 --is_train False --num_eval 1000 --record_video False --init_ckpt_path log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt --record_demo True

# Generate expert trajectories for sub-policy two
python -m run --algo gail --furniture_name chair_ingolf_0650 --demo_path demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 1 --preassembled 0 --run_prefix p1 --is_train False --num_eval 2000 --record_video False --init_ckpt_path log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt --record_demo True

# Generate expert trajectories for fine-tuned sub-policies one and two
python -m run --algo ps --furniture_name chair_ingolf_0650 --ps_demo_paths demos/chair_ingolf/Sawyer_chair_ingolf_0650_0,demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 --num_connects 2 --run_prefix fix2 --is_train False --num_eval 1000 --record_video False --ps_ckpts log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt,log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt --init_ckpt_path log/chair_ingolf_0650.ps.fix2.123/ckpt_00002048000.pt --seed=1234 --record_demo True
```

## Visual Behavior Cloning Policy

### Method One
```
# Train policy one
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--subtask_id=0 \
--demo_path=demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 \
--run_prefix=p0 \
--algo=gail \
--run_name_postfix=policy1_fixed \
--wandb=True

# Train policy two
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--subtask_id=1 \
--demo_path=demos/chair_ingolf/Sawyer_chair_ingolf_0650_1 \
--run_prefix=p1 \
--algo=gail \
--preassembled=0 \
--run_name_postfix=policy2_fixed_finetuned \
--wandb=True

# Evaluate policies one and two
python bc.py \
--is_eval=True \
--eval_mode=two_policy \
--checkpoint=BC_04.02.10.44.25_policy1_fixed/epoch_227.pth \
--p2_checkpoint=BC_04.02.15.53.31_policy2_fixed_finetuned/epoch_32.pth \
--num_eval_eps=100 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2
```

### Method Two
```
# Train policy one
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--subtask_id=0 \
--demo_path=demos/chair_ingolf/Sawyer_chair_ingolf_0650_0 \
--run_prefix=p0 \
--algo=gail \
--run_name_postfix=policy1_fixed \
--wandb=True

# Train policy two
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--subtask_id=1 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2 \
--train_mode=two_policy \
--p1_checkpoint=BC_04.02.10.44.25_policy1_fixed/epoch_227.pth \
--run_name_postfix=policy2_fixed_finetuned_frozenp1val \
--wandb=True

# Evaluate policies one and two
python bc.py \
--is_eval=True \
--eval_mode=two_policy \
--checkpoint=BC_04.02.10.44.25_policy1_fixed/epoch_227.pth \
--p2_checkpoint=BC_04.03.12.02.30_policy2_fixed_finetuned_frozenp1val/epoch_853.pth \
--num_eval_eps=100 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2
```

### Method Three
```
# Train a single BC policy
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2 \
--run_name_postfix=ps_fixed_finetuned_distilled2policies \
--wandb=True

# Evaluate the single BC policy
python bc.py \
--is_eval=True \
--checkpoint=BC_04.03.13.49.24_ps_fixed_finetuned_distilled2policies/epoch_401.pth \
--num_eval_eps=100 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2
```

### Method Four
```
# Jointly train BC policies one and two
python bc.py \
--bc_data=bc_data/chair_ingolf_0650.ps.fix2.1234_step_00002048000_1000_trajs.pkl \
--batch_size=128 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2 \
--train_mode=joint_training \
--run_name_postfix=jointtrain_p1andp2 \
--wandb=True

# Evaluate the BC policies
python bc.py \
--is_eval=True \
--train_mode=joint_training \
--checkpoint=BC_04.03.15.17.19_jointtrain_p1andp2/epoch_304.pth \
--num_eval_eps=100 \
--demo_path=None \
--run_prefix=ours \
--algo=ps \
--num_connects=2
```

---


# Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization

[[Project website](https://clvrai.com/skill-chaining)] [[Paper](https://openreview.net/forum?id=K5-J-Espnaq)]

This project is a PyTorch implementation of [Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization](https://clvrai.com/skill-chaining), published in CoRL 2021.


### Note that Unity rendering for IKEA Furniture Assembly Environment is temporally not available due to the deprecated Unity-MuJoCo plugin in the new version of MuJoCo (2.1). It is still working with MuJoCo 2.0.


## Files and Directories
* `run.py`: launches an appropriate trainer based on algorithm
* `policy_sequencing_trainer.py`: trainer for policy sequencing
* `policy_sequencing_agent.py`: model and training code for policy sequencing
* `policy_sequencing_rollout.py`: rollout with policy sequencing agent
* `policy_sequencing_config.py`: hyperparameters
* `method/`: implementation of IL and RL algorithms
* `furniture/`: IKEA furniture environment
* `demos/`: default demonstration directory
* `log/`: default training log directory
* `result/`: evaluation result directory


## Prerequisites
* Ubuntu 18.04 or above
* Python 3.6
* Mujoco 2.1


## Installation

0. Clone this repository and submodules.
```bash
$ git clone --recursive git@github.com:clvrai/skill-chaining.git
```

1. Install mujoco 2.1 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
Note that the code is compatible with **MuJoCo 2.0**, which supports Unity rendering.
```bash
# download mujoco 2.1
$ mkdir ~/.mujoco
$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco_linux.tar.gz
$ tar -xvzf mujoco_linux.tar.gz -C ~/.mujoco/
$ rm mujoco_linux.tar.gz

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

2. Install python dependencies
```bash
$ sudo apt-get install cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev

# software rendering
$ sudo apt-get install libgl1-mesa-glx libosmesa6 patchelf

# window rendering
$ sudo apt-get install libglfw3 libglew2.0
```

3. Install furniture submodule
```bash
$ cd furniture
$ pip install -e .
$ cd ../method
$ pip install -e .
$ pip install torch torchvision
```


## Usage

For `chair_ingolf_0650`, simply change `table_lack_0825` to `chair_ingolf_0650` in the commands. For training with gpu, specify the desired gpu number (e.g. `--gpu 0`). To change the random seed, append, e.g., `--seed 0` to the command.

To enable wandb logging, add the following arguments with your wandb entity and project names: `--wandb True --wandb_entity [WANDB ENTITY] --wandb_project [WANDB_PROJECT]`.


1. Generate demos
```
# Sub-task demo generation
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --start_count 0 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0 --start_count 1000 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1 --start_count 2000 --phase_ob True
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack/ --reset_robot_after_attach True --max_episode_steps 200 --num_connects 1 --n_demos 200 --preassembled 0,1,2 --start_count 3000 --phase_ob True

# Full-task demo generation
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --demo_dir demos/table_lack_full/ --reset_robot_after_attach True --max_episode_steps 800 --num_connects 4 --n_demos 200 --start_count 0 --phase_ob True
```

2. Train sub-task policies
```
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_0 --num_connects 1 --run_prefix p0
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_1 --num_connects 1 --preassembled 0 --run_prefix p1 --load_init_states log/table_lack_0825.gail.p0.123/success_00024576000.pkl
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_2 --num_connects 1 --preassembled 0,1 --run_prefix p2 --load_init_states log/table_lack_0825.gail.p1.123/success_00030310400.pkl
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_3 --num_connects 1 --preassembled 0,1,2 --run_prefix p3 --load_init_states log/table_lack_0825.gail.p2.123/success_00027852800.pkl
```

3. Collect successful terminal states from sub-task policies
Find the best performing checkpoint from WandB, and replace checkpoint path with the best performing checkpoint (e.g. `--init_ckpt_path log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt`).
```
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_0 --num_connects 1 --run_prefix p0 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p0.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_1 --num_connects 1 --preassembled 0 --run_prefix p1 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p1.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_2 --num_connects 1 --preassembled 0,1 --run_prefix p2 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p2.123/ckpt_00000000000.pt
python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack/Sawyer_table_lack_0825_3 --num_connects 1 --preassembled 0,1,2 --run_prefix p3 --is_train False --num_eval 200 --record_video False --init_ckpt_path log/table_lack_0825.gail.p3.123/ckpt_00000000000.pt
```

4. Train skill chaining
Use the best performing checkpoints (`--ps_ckpt`) and their successful terminal states (`--ps_laod_init_states`).
```
# Ours
mpirun -np 16 python -m run --algo ps --furniture_name table_lack_0825 --num_connects 4 --run_prefix ours \
--ps_ckpts log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p1.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p2.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p3.123/ckpt_00021299200.pt \
--ps_load_init_states log/table_lack_0825.gail.p0.123/success_00021299200.pkl,log/table_lack_0825.gail.p1.123/success_00021299200.pkl,log/table_lack_0825.gail.p2.123/success_00021299200.pkl,log/table_lack_0825.gail.p3.123/success_00021299200.pkl \
--ps_demo_paths demos/table_lack/Sawyer_table_lack_0825_0,demos/table_lack/Sawyer_table_lack_0825_1,demos/table_lack/Sawyer_table_lack_0825_2,demos/table_lack/Sawyer_table_lack_0825_3

# Policy Sequencing (Clegg et al. 2018)
mpirun -np 16 python -m run --algo ps --furniture_name table_lack_0825 --num_connects 4 --run_prefix ps \
--ps_ckpts log/table_lack_0825.gail.p0.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p1.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p2.123/ckpt_00021299200.pt,log/table_lack_0825.gail.p3.123/ckpt_00021299200.pt \
--ps_load_init_states log/table_lack_0825.gail.p0.123/success_00021299200.pkl,log/table_lack_0825.gail.p1.123/success_00021299200.pkl,log/table_lack_0825.gail.p2.123/success_00021299200.pkl,log/table_lack_0825.gail.p3.123/success_00021299200.pkl \
--ps_demo_paths demos/table_lack/Sawyer_table_lack_0825_0,demos/table_lack/Sawyer_table_lack_0825_1,demos/table_lack/Sawyer_table_lack_0825_2,demos/table_lack/Sawyer_table_lack_0825_3
```

5. Train baselines
```
# BC
python -m run --algo bc --max_global_step 1000 --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --record_video False --run_prefix bc --gpu 0

# GAIL
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix gail --gail_env_reward 0

# GAIL+PPO
mpirun -np 16 python -m run --algo gail --furniture_name table_lack_0825 --demo_path demos/table_lack_full/Sawyer_table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix gail_ppo

# PPO
mpirun -np 16 python -m run --algo ppo --furniture_name table_lack_0825 --num_connects 4 --max_episode_steps 800 --max_global_step 200000000 --run_prefix ppo
```


## Citation
If you find this useful, please cite
```
@inproceedings{lee2021adversarial,
  title={Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization},
  author={Youngwoon Lee and Joseph J. Lim and Anima Anandkumar and Yuke Zhu},
  booktitle={Conference on Robot Learning},
  year={2021},
}
```


## References
- This code is based on Youngwoon's robot-learning repo: https://github.com/youngwoon/robot-learning
- IKEA Furniture Assembly Environment: https://github.com/clvrai/furniture
