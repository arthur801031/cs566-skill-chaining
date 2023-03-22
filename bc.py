import copy
import glob
import os
import time
import random
import math

import pickle
import wandb
import shutil
from collections import deque
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import torchvision

import moviepy.editor as mpy
from termcolor import colored

import tqdm

# require for evaluation
import sys
import env
from tqdm import trange
import argparse
from collections import OrderedDict

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(args.num_threads)

torch.multiprocessing.set_sharing_strategy('file_system') # for RuntimeError: Too many open files. Communication with the workers is no longer possible.

class BC_Visual_Policy_Stochastic(nn.Module):
    def __init__(self, robot_state=0, num_classes=256, img_size=128):
        super(BC_Visual_Policy_Stochastic, self).__init__()

        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(first_linear_layer_size + robot_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

        self._activation_fn = getattr(F, 'relu')

        self.fc_means = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        self.fc_log_stds = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # Defining the forward pass
    def forward(self, x, robot_state):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        # concatenate img features with robot_state's information
        x = torch.cat([x, robot_state], dim=1)
        x = self.linear_layers(x)

        x = self._activation_fn(x)

        means = self.fc_means(x)
        log_std = self.fc_log_stds(x)
        log_std = torch.clamp(log_std, -10, 2)
        stds = torch.exp(log_std.double())
        means = OrderedDict([("default", means)])
        stds = OrderedDict([("default", stds)])

        z = FixedNormal(means['default'], stds['default']).rsample()

        action = torch.tanh(z)

        return action

class StateImageActionDataset(Dataset):
    def __init__(self, config, pickle_file, transform=None):
        self.config = config
        rollout_file = open(pickle_file, 'rb')
        self.data = pickle.load(rollout_file)
        rollout_file.close()
        self.transform = transform

    def __len__(self):
        return len(self.data['obs'])

    def random_crop_and_pad(self, img, crop=84):
        """
            source: https://github.com/MishaLaskin/rad/blob/master/data_augs.py
            args:
            img: np.array shape (C,H,W)
            crop: crop size (e.g. 84)
            returns np.array
        """
        data_aug_prob = random.uniform(0, 1)
        if data_aug_prob < 0.5:
            c, h, w = img.shape
            crop_max = h - crop + 1
            w1 = np.random.randint(0, crop_max)
            h1 = np.random.randint(0, crop_max)
            cropped = np.zeros((c, h, w), dtype=img.dtype)
            cropped[:, h1:h1 + crop, w1:w1 + crop] = img[:, h1:h1 + crop, w1:w1 + crop]
            return cropped
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, ac, img_filepath = self.data["obs"][idx], self.data["acs"][idx], self.data['imgs'][idx]
        img = np.load('../' + img_filepath)

        # customized data augmentations
        if self.config.img_aug :
            if self.config.random_crop:
                img = self.random_crop_and_pad(img, self.config.random_crop_size)

        ob = torch.from_numpy(ob)
        ac = torch.from_numpy(ac)
        img = torch.from_numpy(img)

        # torchvision style data augmentations
        if self.transform:
            img = self.transform(img)

        out = {'ob': ob, 'ac': ac, 'img': img}
        return out

class Evaluation:
    def __init__(self, bc_visual_args, args_mopa):
        os.environ["DISPLAY"] = ":1"
        self.eval_seeds = [52, 93, 156, 377, 1000, 1234]
        self.args_mopa = args_mopa
        self.args_mopa.gpu= int(bc_visual_args.cuda_num)
        self.bc_visual_args = bc_visual_args
        self.model = bc_visual_args.model
        self.robot_state_size = bc_visual_args.robot_state_size
        self.action_size = bc_visual_args.action_size
        self._env_eval = gym.make(self.args_mopa.env, **self.args_mopa.__dict__)

    def get_img_robot_state(self, obs):
        obs_img = torch.from_numpy(obs['image'])
        if self.args_mopa.env == 'PusherObstacle-v0':
            state_info = list(obs.values())
            state_info = state_info[0:2]
            obs_robot = retrieve_np_state(state_info)
        elif self.args_mopa.env == 'SawyerPushObstacle-v0' or \
            self.args_mopa.env == 'SawyerAssemblyObstacle-v0' or \
            self.args_mopa.env == 'SawyerLiftObstacle-v0':
            obs_robot = np.concatenate((obs['joint_pos'], obs['joint_vel'], obs['gripper_qpos'], obs['gripper_qvel'], obs['eef_pos'], obs['eef_quat']))

        obs_robot = torch.from_numpy(obs_robot).float()
        obs_robot = obs_robot[None, :]
        return obs_img.cuda(), obs_robot.cuda()

    def evaluate(self, checkpoint):
        policy_eval = BC_Visual_Policy_Stochastic(robot_state=self.robot_state_size, num_classes=self.action_size, img_size=self.bc_visual_args.env_image_size)

        policy_eval.cuda()
        policy_eval.load_state_dict(checkpoint['state_dict'])
        policy_eval.eval()

        num_ep = self.bc_visual_args.num_eval_ep_validation_per_seed

        total_success, total_rewards = 0, 0
        for eval_seed in tqdm.tqdm(self.eval_seeds):
            self._env_eval.set_seed(eval_seed)
            print("\n", colored("Running seed {}".format(eval_seed), "blue"))
            for ep in range(num_ep):
                obs = self._env_eval.reset()

                obs_img, obs_robot = self.get_img_robot_state(obs)

                done = False
                ep_len = 0
                ep_rew = 0

                while ep_len < self.bc_visual_args.eval_bc_max_step_validation and not done:
                    action = policy_eval(obs_img, obs_robot)

                    if len(action.shape) == 2:
                        action = action[0]
                    obs, reward, done, info = self._env_eval.step(action.detach().cpu().numpy(), is_bc_policy=True)
                    obs_img, obs_robot = self.get_img_robot_state(obs)
                    ep_len += 1
                    ep_rew += reward
                    if(ep_len % 100 == 0):
                        print(colored("Current Episode Step: {}, Reward: {}".format(ep_len, reward), "green"))

                print(colored("Current Episode Total Rewards: {}".format(ep_rew), "yellow"))
                if self._env_eval._success:
                    print(colored("Success!", "yellow"), "\n")
                    total_success += 1
                total_rewards += ep_rew
        del policy_eval
        return total_success, total_rewards

# def retrieve_np_state(raw_state):
#     for idx, values in enumerate(raw_state):
#         if(idx==0):
#             ot = np.array(values)
#         else:
#             ot = np.concatenate((ot, np.array(values)), axis=0)

#     return ot

# def overwrite_env_args(env_args):
#     env_args.env = args.env
#     env_args.env_image_size = args.env_image_size
#     env_args.seed = args.env_seed
#     env_args.screen_width = args.screen_width
#     env_args.screen_height = args.screen_height
#     env_args.obs_space = 'all'

# def get_mopa_rl_agent(args_mopa, device, ckpt):
#     mopa_config = copy.deepcopy(args_mopa)
#     mopa_config.policy = 'mlp'
#     mopa_config.obs_space = 'state'
#     expert_actor, expert_critic = get_actor_critic_by_name(mopa_config.policy)

#     ckpt = torch.load(ckpt)
#     env = gym.make(mopa_config.env, **mopa_config.__dict__)
#     ob_space = env.observation_space
#     ac_space = env.action_space

#     critic1 = expert_critic(mopa_config, ob_space, ac_space)    
#     critic1.load_state_dict(ckpt["agent"]["critic1_state_dict"])
#     critic1.to(device)
#     critic1.eval()

#     critic2 = expert_critic(mopa_config, ob_space, ac_space)    
#     critic2.load_state_dict(ckpt["agent"]["critic2_state_dict"])
#     critic2.to(device)
#     critic2.eval()

#     return critic1, critic2

def main():
    parser = argparse.ArgumentParser()

    ## training
    parser.add_argument('--start_epoch', type=int, default=0, help="starting epoch for training")
    parser.add_argument('--end_epoch', type=int, default=1000, help="ending epoch for training")
    parser.add_argument('--lrate', type=float, default=0.0005, help="initial learning rate for the policy network update")
    parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
    parser.add_argument('--beta2', type=float, default=0.9, help="betas for Adam Optimizer")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size for model training")
    parser.add_argument('--load_saved', type=bool, default=False, help="load weights from the saved model")
    parser.add_argument('--model_save_dir', type=str, default='../checkpoints/sawyer_assembly_32px_0.4million_checkpoint', help="directory for saving trained model weights")
    parser.add_argument('--checkpoint', type=str, default='epoch_12.pth', help="checkpoint file")
    parser.add_argument('--saved_rollouts', type=str, default='../saved_rollouts', help="directory to load saved expert demonstrations from")
    parser.add_argument('--saved_rollouts_file', type=str, default='../saved_rollouts/sawyer-assembly-21files-0.4million/combined.pickle', help="file to load saved expert demonstrations from")
    parser.add_argument('--saved_rollouts_vis', type=str, default='../saved_rollouts', help="directory to save visualization of the covered states from saved bc data")
    parser.add_argument('--seed', type=int, default=1234, help="torch seed value")
    parser.add_argument('--num_threads', type=int, default=1, help="number of threads for execution")
    parser.add_argument('--train_data_ratio', type=float, default=0.90, help="ratio for training data for train-test split")
    parser.add_argument("--model", type=str, default="BC_Visual_Policy_Stochastic", choices=["BC_Visual_Policy", "BC_Image_Only", "BC_Robot_Only", "BC_Visual_Policy_Stochastic", "BC_Visual_Stochastic_w_Critics"], help="choice of model")

    ## data augmentation
    parser.add_argument('--img_aug', type=bool, default=False, help="whether to use data augmentations on images")
    # random crop
    parser.add_argument('--random_crop', type=bool, default=True, help="whether to use random crop")
    parser.add_argument('--random_crop_size', type=int, default=24, help="random crop size")

    ## scheduler
    parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")

    ## cuda
    parser.add_argument('--cuda_num', type=str, default='1', help="use gpu for computation")

    ## logs
    parser.add_argument('--wandb', type=bool, default=True, help="learning curves logged on weights and biases")
    parser.add_argument('--print_iteration', type=int, default=1000, help="iteration interval for displaying current loss values")

    ## validation arguments
    parser.add_argument('--num_eval_ep_validation_per_seed', type=int, default=5, help="number of episodes to run during evaluation")
    parser.add_argument('--eval_bc_max_step_validation', type=int, default=400, help="maximum steps during evaluations of learnt bc policy")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluation_interval")

    ## bc args
    parser.add_argument('--image_rollouts', type=bool, default=False, help="whether the bc observations are state based or image based")
    parser.add_argument('--stacked_states', type=bool, default=False, help="whether to use stacked frames as observations or individually")
    parser.add_argument('--num_stack_frames', type=int, default=4, help="number of frames to be stacked for each observation")
    parser.add_argument('--action_size', type=int, default=4, help="dimension of the action space")
    parser.add_argument('--robot_state_size', type=int, default=14, help="dimension of the observation space")
    parser.add_argument('--env_state_size', type=int, default=0, help="dimension of the environment space")
    parser.add_argument('--bc_video_dir', type=str, default='../bc_visual_videos', help="directory to store behavioral cloning video simulations")
    parser.add_argument('--eval_bc_max_step', type=int, default=1000, help="maximum steps during evaluations of learnt bc policy")
    parser.add_argument('--num_eval_ep', type=int, default=100, help="number of episodes to run during evaluation")
    parser.add_argument('--three_hundred_eval_five_seeds', type=bool, default=True, help="500 evaluations (100 for each random seed [1234, 200, 500, 2320, 1800])")
    parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor for calculating discounted rewards")

    args = parser.parse_args()

    torch.set_num_threads(1)
    global val_loss_best
    val_loss_best = 1e10

    device = torch.device("cuda:0")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)

    mse_loss = nn.MSELoss()

    # image augmentations
    transform = None
    # torchvision data augmentations style
    # if args.img_aug:
    #     img_augs = [] 
    #     if args.random_crop:
    #         img_augs.append(torchvision.transforms.RandomCrop(size=args.random_crop_size))
    #         args.env_image_size = args.random_crop_size
    #     transform = torchvision.transforms.Compose(img_augs)
    #     print('Applying data augmentations on images...')

    policy = BC_Visual_Policy_Stochastic(robot_state=args.robot_state_size, num_classes=args.action_size, img_size=args.env_image_size)

    if args.wandb:
        wandb.init(
            project="skill-chaining",
            config={k: v for k, v in args.__dict__.items()}
        )
        wandb.watch(policy)

    print(colored('Training model {}'.format(args.model), 'blue'))
    print(policy)

    if args.model == 'BC_Visual_Stochastic_w_Critics':
        critic1_optimizer = optim.Adam(list(critic1.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
        critic2_optimizer = optim.Adam(list(critic2.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
        scheduler_critic1 = optim.lr_scheduler.StepLR(critic1_optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        scheduler_critic2 = optim.lr_scheduler.StepLR(critic2_optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    optimizer = optim.Adam(list(policy.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # load from checkpoint
    if args.load_saved:
        checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint), map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        policy.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # fix for bug: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336031198
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        scheduler.load_state_dict(checkpoint['scheduler'])
        val_loss_best = checkpoint['val_loss_best']

        if args.model == 'BC_Visual_Stochastic_w_Critics':
            critic1.load_state_dict(checkpoint['critic1_state_dict'])
            critic2.load_state_dict(checkpoint['critic2_state_dict'])
    else:
        start_epoch = args.start_epoch

    dataset = StateImageActionDataset(args, args.saved_rollouts_file, transform=transform)
    dataset_length = len(dataset)
    train_size = int(args.train_data_ratio * dataset_length)
    test_size = dataset_length - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    train_loss = []
    val_loss = []

    policy.cuda()
    mse_loss.cuda()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    evaluation_obj = Evaluation(args, args_mopa)

    print('Total number of state-action pairs: ', dataset_length)
    print('Number of training state-action pairs: ', len(train_dataset))
    print('Number of test state-action pairs: ', len(test_dataset))
    outer = tqdm.tqdm(total=args.end_epoch-start_epoch, desc='Epoch', position=start_epoch)
    for epoch in range(start_epoch, args.end_epoch):
        total_loss = 0.0
        validation_loss = 0.0

        policy.train()

        print('\nprocessing training batch...')
        for i, batch in enumerate(dataloader_train):
            ob, ac, img = batch['ob'], batch['ac'], batch['img']
            ob = ob.float().cuda()
            ac = ac.float().cuda()
            img = img.float().cuda()

            ac_pred = policy(img, ob)
            # ac mse
            ac_predictor_loss = mse_loss(ac_pred, ac)
            optimizer.zero_grad()
            ac_predictor_loss.backward()
            optimizer.step()
            total_loss += ac_predictor_loss.data.item()

        training_loss = total_loss / (args.batch_size*len(dataloader_train))
        train_loss.append(training_loss)

        print('')
        print('----------------------------------------------------------------------')
        print('Epoch #' + str(epoch))
        print('Action Prediction Loss (Train): ' + str(training_loss))
        print('----------------------------------------------------------------------')

        # evaluating on test set
        policy.eval()

        action_predictor_loss_val = 0.

        print('\nprocessing test batch...')
        for i, batch in enumerate(dataloader_test):
            ob, ac, img = batch['ob'], batch['ac'], batch['img']
            ob = ob.float().cuda()
            ac = ac.float().cuda()
            img = img.float().cuda()
            ac_pred = policy(img, ob)

            action_predictor_loss_val = mse_loss(ac_pred, ac)
            validation_loss += action_predictor_loss_val.data.item()

        validation_loss /= (args.batch_size * len(dataloader_test))
        val_loss.append(validation_loss)

        print('')
        print('**********************************************************************')
        print('Epoch #' + str(epoch))
        print('')
        print('Action Prediction Loss (Test): ' + str(validation_loss))
        print()
        print('**********************************************************************')

        scheduler.step()
        if(validation_loss<val_loss_best):
            val_loss_best = validation_loss
            print(colored("BEST VAL LOSS: {}".format(val_loss_best), "yellow"))

        # arrange/save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss_best': val_loss_best,
        }
        torch.save(checkpoint, os.path.join(args.model_save_dir, 'epoch_{}.pth'.format(epoch)))

        # perform validation
        if epoch % args.eval_interval == 0:
            total_success, total_rewards = evaluation_obj.evaluate(checkpoint)
        else:
            total_success, total_rewards = -1, -1

        # wandb logging
        if args.wandb:
            wandb.log({
                "Epoch": epoch,
                "Total Success": total_success,
                "Total Rewards": total_rewards,
                "Action Prediction Loss (Train)": training_loss,
                "Action Prediction Loss (Test)": validation_loss,
            })
        else:
            plt.plot(train_loss, label="train loss")
            plt.plot(val_loss, label="validation loss")
            plt.legend()
            plt.savefig(os.path.join(args.bc_video_dir, 'train_loss_plots.png'))
            plt.close()
        outer.update(1)

if __name__ == "__main__":
    main()