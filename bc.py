import os
import random
import math
import cv2
import pickle
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.multiprocessing
import matplotlib.pyplot as plt
import torchvision
from termcolor import colored

from tqdm import tqdm
from datetime import datetime

# require for evaluation
import sys
import argparse
from collections import OrderedDict
from robot_learning.environments import make_env
import imageio

class BC_Visual_Policy_Stochastic(nn.Module):
    def __init__(self, action_size=256, img_size=128):
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
            nn.Linear(first_linear_layer_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

        self._activation_fn = getattr(F, 'relu')

        self.fc_means = nn.Sequential(
            nn.Linear(256, action_size),
        )

        self.fc_log_stds = nn.Sequential(
            nn.Linear(256, action_size),
        )

        self.double()

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
    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)

        x = self._activation_fn(x)

        means = self.fc_means(x)
        log_std = self.fc_log_stds(x)
        log_std = torch.clamp(log_std, -10, 2)
        stds = torch.exp(log_std.double())
        means = OrderedDict([("default", means)])
        stds = OrderedDict([("default", stds)])

        z = torch.distributions.Normal(means['default'], stds['default']).rsample()

        action = torch.tanh(z)

        return action

class StateImageActionDataset(Dataset):
    def __init__(self, config, pickle_file, transform=None):
        self.config = config
        self.load_data(pickle_file)
        self.transform = transform

    def __len__(self):
        return len(self.data['ob_images'])

    def load_data(self, pickle_file):
        rollout_file = open(pickle_file, 'rb')
        tmp_data = pickle.load(rollout_file)
        rollout_file.close()
        self.data = {
            'ob_images': [],
            'actions': [],
        }
        for rollout in tmp_data:
            if self.config.subtask_id == -1:
                for ob_image, action in zip(rollout['ob_images'], rollout['actions']):
                    ob_image = cv2.resize(ob_image, (self.config.env_image_size, self.config.env_image_size))
                    self.data['ob_images'].append(np.transpose(ob_image, (2, 0, 1)))
                    self.data['actions'].append(action['default'])
            else:
                for ob_image, action, subtask_id in zip(rollout['ob_images'], rollout['actions'], rollout['subtasks']):
                    if self.config.subtask_id == subtask_id:
                        # # DEBUG observation image
                        # sample_img = ob_image
                        # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite('./input_image_bc_load_test_subtask1.png', sample_img)
                        # import pdb; pdb.set_trace()
                        ob_image = cv2.resize(ob_image, (self.config.env_image_size, self.config.env_image_size))
                        self.data['ob_images'].append(np.transpose(ob_image, (2, 0, 1)))
                        self.data['actions'].append(action['default'])

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

        img = self.data['ob_images'][idx]
        ac = self.data['actions'][idx]

        # customized data augmentations: we don't need this because we're using self.transform
        # if self.config.img_aug :
        #     if self.config.random_crop:
        #         img = self.random_crop_and_pad(img, self.config.random_crop_size)

        ac = torch.from_numpy(ac)
        img = torch.from_numpy(img)

        # torchvision style data augmentations
        if self.transform:
            img = self.transform(img)

        out = {'ac': ac, 'img': img}
        return out

class StateImageActionJointTrainingDataset(Dataset):
    def __init__(self, config, pickle_file, transform=None):
        self.config = config
        self.load_data(pickle_file)
        self.transform = transform

    def __len__(self):
        return min(len(self.data['p1_ob_images']), len(self.data['p2_ob_images']))

    def load_data(self, pickle_file):
        rollout_file = open(pickle_file, 'rb')
        tmp_data = pickle.load(rollout_file)
        rollout_file.close()
        self.data = {
            'p1_ob_images': [],
            'p1_actions': [],
            'p2_ob_images': [],
            'p2_actions': [],
        }
        for rollout in tmp_data:
            for ob_image, action, subtask_id in zip(rollout['ob_images'], rollout['actions'], rollout['subtasks']):
                ob_image = cv2.resize(ob_image, (self.config.env_image_size, self.config.env_image_size))
                if subtask_id == 0:
                    # # DEBUG observation image
                    # sample_img = ob_image
                    # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite('./input_image_bc_load_test_subtask1.png', sample_img)
                    # import pdb; pdb.set_trace()
                    self.data['p1_ob_images'].append(np.transpose(ob_image, (2, 0, 1)))
                    self.data['p1_actions'].append(action['default'])
                else:
                    # subtask_id == 1
                    self.data['p2_ob_images'].append(np.transpose(ob_image, (2, 0, 1)))
                    self.data['p2_actions'].append(action['default'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        p1_img = self.data['p1_ob_images'][idx]
        p1_ac = self.data['p1_actions'][idx]
        p1_img = torch.from_numpy(p1_img)
        p1_ac = torch.from_numpy(p1_ac)

        p2_img = self.data['p2_ob_images'][idx]
        p2_ac = self.data['p2_actions'][idx]
        p2_img = torch.from_numpy(p2_img)
        p2_ac = torch.from_numpy(p2_ac)

        out = {'p1_ac': p1_ac, 'p1_img': p1_img, 'p2_ac': p2_ac, 'p2_img': p2_img}
        return out

class Evaluation:
    def __init__(self, args, policy_input_image_size, transform=None):
        self.args = args
        self.policy_input_image_size = policy_input_image_size
        self.transform = transform

        # default arguments from skill-chaining
        from policy_sequencing_config import create_skill_chaining_parser
        parser = create_skill_chaining_parser()
        self._config, unparsed = parser.parse_known_args()

        # set environment specific parameters
        setattr(self._config, 'algo', args.algo)
        setattr(self._config, 'furniture_name', args.furniture_name)
        setattr(self._config, 'env', 'IKEASawyerDense-v0')
        setattr(self._config, 'demo_path', args.demo_path)
        setattr(self._config, 'num_connects', args.num_connects)
        setattr(self._config, 'run_prefix', args.run_prefix)
        setattr(self._config, 'is_train', False)
        setattr(self._config, 'record_video', False)
        if args.preassembled >= 0:
            preassembled = [i for i in range(0, args.preassembled+1)]
            setattr(self._config, 'preassembled', preassembled)
        if args.algo == 'ps':
            # specific parameters for ps algorithm
            # ARTHUR: hard-coded values
            ps_demo_paths = ['demos/chair_ingolf/Sawyer_chair_ingolf_0650_0', 'demos/chair_ingolf/Sawyer_chair_ingolf_0650_1']
            setattr(self._config, 'ps_demo_paths', ps_demo_paths)
            ps_ckpts = ['log/chair_ingolf_0650.gail.p0.123/ckpt_00011468800.pt', 'log/chair_ingolf_0650.gail.p1.123/ckpt_00017203200.pt']
            setattr(self._config, 'ps_ckpts', ps_ckpts)
            setattr(self._config, 'init_ckpt_path', 'log/chair_ingolf_0650.ps.ours.123/ckpt_00010649600.pt')

        if args.is_eval:
            # prepare video directory
            self.p2_ckpt_name = args.p2_checkpoint.split('/')[-1].split('.')[0]
            self.videos_dir = '/'.join(os.path.join(args.model_save_dir, args.p2_checkpoint).split('/')[:-1]) + '/videos'
            if not os.path.exists(self.videos_dir):
                os.makedirs(self.videos_dir)

            if args.eval_mode == 'two_policy':
                self.p2_checkpoint = torch.load(os.path.join(args.model_save_dir, args.p2_checkpoint), map_location='cuda')
        else:
            if args.train_mode == 'two_policy':
                self.policy1_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
                self.policy1_eval.cuda()
                self.policy1_eval.load_state_dict(torch.load(os.path.join(args.model_save_dir, args.p1_checkpoint), map_location='cuda')['state_dict'])
                self.policy1_eval.eval()
                # freeze all layers
                for param in self.policy1_eval.parameters():
                    param.requires_grad = False

        self._env = make_env(self._config.env, self._config)

    def get_ob_image(self, env):
        ob_image = env.render("rgb_array")
        if len(ob_image.shape) == 4:
            ob_image = ob_image[0]
        if np.max(ob_image) <= 1.0:
            ob_image *= 255.0
        ob_image = ob_image.astype(np.uint8)

        # # DEBUG observation image
        # sample_img = ob_image
        # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./input_image_bc_eval_test_p2.png', sample_img)
        # import pdb; pdb.set_trace()

        ob_image = cv2.resize(ob_image, (self.args.env_image_size, self.args.env_image_size))
        ob_image = np.transpose(ob_image, (2, 0, 1))
        ob_image = torch.from_numpy(ob_image).double().cuda()
        if self.transform:
            ob_image = self.transform(ob_image)

        return ob_image[None, :, :, :]

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self.videos_dir, fname)

        if np.issubdtype(frames[0].dtype, np.floating):
            for i in range(len(frames)):
                frames[i] = frames[i].astype(np.uint8)
        imageio.mimsave(path, frames, fps=fps)
        print(f'Video saved: {path}')
        return path

    def _store_frame(self, env, ep_len, ep_rew, info={}):
        """ Renders a frame. """
        color = (200, 200, 200)

        # render video frame
        frame = env.render("rgb_array")
        if len(frame.shape) == 4:
            frame = frame[0]
        if np.max(frame) <= 1.0:
            frame *= 255.0
        frame = frame.astype(np.uint8)

        h, w = frame.shape[:2]
        if h < 512:
            h, w = 512, 512
            frame = cv2.resize(frame, (h, w))
        frame = np.concatenate([frame, np.zeros((h, w, 3))], 0)
        scale = h / 512

        # add caption to video frame
        if self._config.record_video_caption:
            text = "{:4} {}".format(ep_len, ep_rew)
            font_size = 0.4 * scale
            thickness = 1
            offset = int(12 * scale)
            x, y = int(5 * scale), h + int(10 * scale)
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = "{}: ".format(k)
                (key_width, _), _ = cv2.getTextSize(
                    key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                )

                cv2.putText(
                    frame,
                    key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (66, 133, 244),
                    thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return frame

    def evaluate(self, checkpoint):
        policies = []
        self.policy_list_enable = False
        if self.args.train_mode == 'two_policy':
            policy2_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
            policy2_eval.cuda()
            policy2_eval.load_state_dict(checkpoint['state_dict'])
            policy2_eval.eval()
            policies = [self.policy1_eval, policy2_eval]
            self.policy_list_enable = True
        elif self.args.train_mode == 'joint_training':
            policy1_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
            policy1_eval.cuda()
            policy1_eval.load_state_dict(checkpoint['p1_state_dict'])
            policy1_eval.eval()

            policy2_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
            policy2_eval.cuda()
            policy2_eval.load_state_dict(checkpoint['p2_state_dict'])
            policy2_eval.eval()
            policies = [policy1_eval, policy2_eval]
            self.policy_list_enable = True
        else:
            policy_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
            policy_eval.cuda()
            policy_eval.load_state_dict(checkpoint['state_dict'])
            policy_eval.eval()

            if self.args.eval_mode == 'two_policy':
                policy2_eval = BC_Visual_Policy_Stochastic(action_size=self.args.action_size, img_size=self.policy_input_image_size)
                policy2_eval.cuda()
                policy2_eval.load_state_dict(self.p2_checkpoint['state_dict'])
                policy2_eval.eval()
                policies = [policy_eval, policy2_eval]
                self.policy_list_enable = True

        num_subtask2_success = 0
        num_subtask1_success = 0
        total_success, total_rewards, total_lengths, total_subtasks = 0, 0, 0, 0
        for ep in range(self.args.num_eval_eps):
            obs = self._env.reset()

            done = False
            ep_len = 0
            ep_rew = 0

            record_frames = []
            if self.args.is_eval:
                record_frames.append(self._store_frame(self._env, ep_len, ep_rew))

            subtask = 0
            while not done:
                ob_image = self.get_ob_image(self._env)
                if self.policy_list_enable:
                    action = policies[subtask](ob_image)
                else:
                    action = policy_eval(ob_image)
                if len(action.shape) == 2:
                    action = action[0]

                ob_next, reward, done, info = self._env.step(action.detach().cpu().numpy())
                ep_len += 1
                ep_rew += reward

                if "subtask" in info and subtask != info["subtask"]:
                    print(colored(f'Completed subtask {subtask}', "yellow"))
                    subtask = info["subtask"]

                # terminal/goal condition for policy sequencing algorithm (since we're only training 2 sub-policies)
                if self._config.algo == 'ps' and info['subtask'] == 2:
                    done = True
                    info['episode_success'] = True

                if self.args.is_eval:
                    frame_info = info.copy()
                    record_frames.append(self._store_frame(self._env, ep_len, ep_rew, frame_info))

            print(colored(f"Current Episode Total Rewards: {ep_rew}, Episode Length: {ep_len}", "yellow"))
            if 'episode_success' in info and info['episode_success']:
                print(colored(f"{info['episode_success']}!", "yellow"), "\n")
                total_success += 1
                num_subtask2_success += 1
                num_subtask1_success += 1 # since sub-task 2 is completed, sub-task 1 must have been completed
            else:
                if info["subtask"] == 1:
                    num_subtask1_success += 1
            if self.args.is_eval:
                s_flag = 's' if 'episode_success' in info and info['episode_success'] else 'f'
                self._save_video(f'{self.p2_ckpt_name}_ep_{ep}_{ep_rew}_{subtask}_{s_flag}.mp4', record_frames)
            total_rewards += ep_rew
            total_lengths += ep_len
            total_subtasks += subtask
        # output success rate of sub-tasks 1 and 2
        print(f'Success rate of Sub-task 2: {(num_subtask2_success / self.args.num_eval_eps) * 100}%')
        print(f'Success rate of Sub-task 1: {(num_subtask1_success / self.args.num_eval_eps) * 100}%')
        return total_success, total_rewards, total_lengths, total_subtasks

def main():
    parser = argparse.ArgumentParser()

    ## training
    parser.add_argument('--start_epoch', type=int, default=0, help="starting epoch for training")
    parser.add_argument('--end_epoch', type=int, default=1000, help="ending epoch for training")
    parser.add_argument('--lrate', type=float, default=0.0005, help="initial learning rate for the policy network update")
    parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
    parser.add_argument('--beta2', type=float, default=0.9, help="betas for Adam Optimizer")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size for model training")
    parser.add_argument('--load_saved', type=bool, default=False, help="load weights from the saved model")
    parser.add_argument('--model_save_dir', type=str, default='./bc_checkpoints', help="directory for saving trained model weights")
    parser.add_argument('--checkpoint', type=str, default='epoch_12.pth', help="checkpoint file")
    parser.add_argument('--bc_data', type=str, default='./bc_data/chair_ingolf_0650.gail.p0.123_step_00011468800_2_trajs.pkl', help="file to load saved expert demonstrations from")
    parser.add_argument('--seed', type=int, default=1234, help="torch seed value")
    parser.add_argument('--num_threads', type=int, default=1, help="number of threads for execution")
    parser.add_argument('--subtask_id', type=int, default=-1, help="subtask_id is used to retrieve subtask_id data from demonstrations")
    parser.add_argument('--train_mode', default='one_policy', choices=['one_policy', 'two_policy', 'joint_training'])
    parser.add_argument('--p1_checkpoint', type=str, default='epoch_12.pth', help="policy 1 checkpoint file (frozen but use during training's validation)")

    ## data augmentation
    parser.add_argument('--img_aug', type=bool, default=False, help="whether to use data augmentations on images")
    # random crop
    parser.add_argument('--random_crop', type=bool, default=True, help="whether to use random crop")
    parser.add_argument('--random_crop_size', type=int, default=24, help="random crop size")

    ## scheduler
    parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")

    ## logs
    parser.add_argument('--run_name_postfix', type=str, default=None, help="run_name_postfix")
    parser.add_argument('--wandb', type=bool, default=False, help="learning curves logged on weights and biases")

    ## validation arguments
    parser.add_argument('--num_eval_eps', type=int, default=20, help="number of episodes to run during evaluation")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluation_interval")

    ## evaluation arguments
    parser.add_argument('--is_eval', type=bool, default=False, help="Whether to perform evaluation")
    parser.add_argument('--eval_mode', default='one_policy', choices=['one_policy', 'two_policy'])
    parser.add_argument('--p2_checkpoint', type=str, default='epoch_12.pth', help="policy 2 checkpoint file")

    ## bc args
    parser.add_argument('--env_image_size', type=int, default=200, help="observation image size")
    parser.add_argument('--action_size', type=int, default=9, help="dimension of the action space")

    ## skill-chaining args
    parser.add_argument('--furniture_name', type=str, default='chair_ingolf_0650', help="furniture_name")
    parser.add_argument('--demo_path', type=str, default='demos/chair_ingolf/Sawyer_chair_ingolf_0650_0', help="demo_path")
    parser.add_argument('--run_prefix', type=str, default='p0', help="run_prefix")
    parser.add_argument('--algo', type=str, default='gail', help="algo")
    parser.add_argument('--preassembled', type=int, default=-1, help="preassembled")
    parser.add_argument('--num_connects', type=int, default=1, help="num_connects")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system') # for RuntimeError: Too many open files. Communication with the workers is no longer possible.
    device = torch.device("cuda")
    mse_loss = nn.MSELoss()

    if args.run_name_postfix:
        run_name = f'BC_{datetime.now().strftime("%m.%d.%H.%M.%S")}_{args.run_name_postfix}'
    else:
        run_name = f'BC_{datetime.now().strftime("%m.%d.%H.%M.%S")}'

    # image augmentations
    transform = None
    policy_input_image_size = args.env_image_size
    # torchvision data augmentations style
    if args.img_aug:
        img_augs = []
        if args.random_crop:
            img_augs.append(torchvision.transforms.RandomCrop(size=args.random_crop_size))
            policy_input_image_size = args.random_crop_size
        transform = torchvision.transforms.Compose(img_augs)
        print('Applying data augmentations on images...')

    policy = BC_Visual_Policy_Stochastic(action_size=args.action_size, img_size=policy_input_image_size)
    if args.wandb:
        wandb.init(
            project="skill-chaining",
            name=run_name,
            config={k: v for k, v in args.__dict__.items()}
        )
        wandb.watch(policy)
    print(run_name)
    print(policy)

    optimizer = optim.Adam(list(policy.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    if args.train_mode == 'joint_training':
        policy2 = BC_Visual_Policy_Stochastic(action_size=args.action_size, img_size=policy_input_image_size)
        optimizer2 = optim.Adam(list(policy2.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        policy2.cuda()

    # load from checkpoint
    if args.load_saved or args.is_eval:
        checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint), map_location='cuda')
        if args.load_saved:
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
    else:
        start_epoch = args.start_epoch

    policy.cuda()
    mse_loss.cuda()

    evaluation_obj = Evaluation(args, policy_input_image_size, transform)

    if args.is_eval:
        total_success, total_rewards, total_lengths, total_subtasks = evaluation_obj.evaluate(checkpoint)
        print(f'Success rate: {(total_success / args.num_eval_eps) * 100}%')
        print(f'Average rewards: {total_rewards / args.num_eval_eps}')
        print(f'Average episode length: {total_lengths / args.num_eval_eps}')
        print(f'Average subtasks: {total_subtasks / args.num_eval_eps}')
    else:
        args.model_save_dir = os.path.join(args.model_save_dir, run_name)
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        if args.train_mode == 'joint_training':
            dataset = StateImageActionJointTrainingDataset(args, args.bc_data, transform=transform)
        else:
            dataset = StateImageActionDataset(args, args.bc_data, transform=transform)
        dataset_length = len(dataset)
        train_dataset, _ = torch.utils.data.random_split(dataset, [dataset_length, 0])
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

        train_loss = []

        print('Total number of state-action pairs: ', dataset_length)
        outer = tqdm(total=args.end_epoch-start_epoch, desc='Epoch', position=start_epoch)
        for epoch in range(start_epoch, args.end_epoch):
            total_loss = 0.0
            print('\nprocessing training batch...')

            if args.train_mode == 'joint_training':
                policy.train()
                policy2.train()
                for i, batch in enumerate(tqdm(dataloader_train)):
                    p1_ac, p1_img = batch['p1_ac'], batch['p1_img']
                    p2_ac, p2_img = batch['p2_ac'], batch['p2_img']
                    p1_ac = p1_ac.double().cuda()
                    p1_img = p1_img.double().cuda()
                    p2_ac = p2_ac.double().cuda()
                    p2_img = p2_img.double().cuda()

                    p1_ac_pred = policy(p1_img)
                    p2_ac_pred = policy2(p2_img)

                    p1_ac_predictor_loss = mse_loss(p1_ac_pred, p1_ac)
                    p2_ac_predictor_loss = mse_loss(p2_ac_pred, p2_ac)

                    total_training_loss = p1_ac_predictor_loss + p2_ac_predictor_loss
                    optimizer.zero_grad()
                    optimizer2.zero_grad()
                    total_training_loss.backward()
                    optimizer.step()
                    optimizer2.step()
                    total_loss += total_training_loss.data.item()
                training_loss = total_loss / (args.batch_size*len(dataloader_train)*2)
            else:
                policy.train()
                for i, batch in enumerate(tqdm(dataloader_train)):
                    ac, img = batch['ac'], batch['img']
                    ac = ac.double().cuda()
                    img = img.double().cuda()

                    ac_pred = policy(img)
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

            if args.train_mode == 'joint_training':
                scheduler.step()
                scheduler2.step()
                # arrange/save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'p1_state_dict': policy.state_dict(),
                    'p1_optimizer': optimizer.state_dict(),
                    'p1_scheduler': scheduler.state_dict(),
                    'p2_state_dict': policy2.state_dict(),
                    'p2_optimizer': optimizer2.state_dict(),
                    'p2_scheduler': scheduler2.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.model_save_dir, 'epoch_{}.pth'.format(epoch)))
                policy.eval()
                policy2.eval()
            else:
                scheduler.step()
                # arrange/save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.model_save_dir, 'epoch_{}.pth'.format(epoch)))
                policy.eval()

            # perform validation
            if epoch % args.eval_interval == 0:
                total_success, total_rewards, total_lengths, total_subtasks = evaluation_obj.evaluate(checkpoint)
            else:
                total_success, total_rewards, total_lengths, total_subtasks = -1, -1, -1, -1

            # wandb logging
            if args.wandb:
                wandb.log({
                    "Epoch": epoch,
                    "Total Success": total_success,
                    "Total Rewards": total_rewards,
                    "Total Lengths": total_lengths,
                    "Total Subtasks": total_subtasks,
                    "Action Prediction Loss (Train)": training_loss,
                })
            else:
                plt.plot(train_loss, label="train loss")
                plt.legend()
                plt.savefig(os.path.join(args.model_save_dir, 'train_loss_plots.png'))
                plt.close()
            outer.update(1)

if __name__ == "__main__":
    main()