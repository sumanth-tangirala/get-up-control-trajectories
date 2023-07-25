import argparse
import json
import os
import sys
import time
from functools import partial

import cv2
import imageio
import numpy as np
import torch

import utils
from SAC import SAC
from env import HumanoidStandupEnv, HumanoidStandupVelocityEnv, HumanoidVariantStandupEnv, \
    HumanoidVariantStandupVelocityEnv, TOTAL_STEPS
from utils import RLLogger, ReplayBuffer, organize_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=5, suppress=True)


class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--env', type=str, default='HumanoidStandup',
                          choices=['HumanoidStandup', 'HumanoidVariantStandup'])
        self.add_argument('--variant', type=str, default='', choices=['Disabled', 'Noarm'])
        self.add_argument('--test_policy', default=False, action='store_true')
        self.add_argument('--teacher_student', default=False, action='store_true')
        self.add_argument('--to_file', default=False, action='store_true')
        self.add_argument("--teacher_power", default=0.4, type=float)
        self.add_argument("--teacher_dir", default=None, type=str)
        self.add_argument("--seed", default=0, type=int)
        self.add_argument("--power", default=1.0, type=float)
        self.add_argument("--curr_power", default=1.0, type=float)
        self.add_argument("--power_end", default=0.4, type=float)
        self.add_argument("--slow_speed", default=0.2, type=float)
        self.add_argument("--fast_speed", default=0.8, type=float)
        self.add_argument("--target_speed", default=0.5, type=float, help="Target speed is used to test the weaker policy")
        self.add_argument("--threshold", default=60, type=float)
        self.add_argument('--max_timesteps', type=int, default=10000000, help='Number of simulation steps to run')
        self.add_argument('--test_interval', type=int, default=20000, help='Number of simulation steps between tests')
        self.add_argument('--test_iterations', type=int, default=10, help='Number of test episodes')
        self.add_argument('--replay_buffer_size', type=int, default=1e6, help='Capacity of the replay buffer')
        self.add_argument('--avg_reward', default=False, action='store_true')
        self.add_argument("--work_dir", default='./experiment/')
        self.add_argument("--load_dir", default=None, type=str)
        # SAC hyperparameters
        self.add_argument("--batch_size", default=1024, type=int)
        self.add_argument("--discount", default=0.99, type=float)
        self.add_argument("--init_temperature", default=0.1, type=float)
        self.add_argument("--critic_target_update_freq", default=2, type=int)
        self.add_argument("--alpha_lr", default=1e-4, type=float)
        self.add_argument("--actor_lr", default=1e-4, type=float)
        self.add_argument("--critic_lr", default=1e-4, type=float)
        self.add_argument("--tau", default=0.005)
        self.add_argument("--start_timesteps", default=10000, type=int)
        self.add_argument('--log_interval', type=int, default=100, help='log every N')
        self.add_argument("--tag", default="")
        #New Args
        self.add_argument('--get_trajectories', default=False, action='store_true')
def main():
    args = ArgParserTrain().parse_args()
    if args.get_trajectories:
        get_trajectories(args)
        return
    trainer = Trainer(args)
    trainer.train_sac()

def get_trajectories(args):
    args.test_policy = True
    args.teacher_student = True
    if args.teacher_dir is None:
        args.teacher_dir = "experiment/pretrained/teacher"
    if args.load_dir is None:
        args.load_dir = "experiment/pretrained/student"

    args = ArgParserTrain().parse_args(namespace=args)

    exp_dir = utils.generate_trajectories_dir(args)
    utils.make_dir(os.path.join(exp_dir, 'videos'))
    utils.make_dir(os.path.join(exp_dir, 'state'))
    utils.make_dir(os.path.join(exp_dir, 'success_params'))

    generate_trajectory_partial = partial(generate_trajectory, exp_dir=exp_dir, args=args)

    utils.parallel_run(generate_trajectory_partial, list(range(args.test_iterations)), should_return=False)

def get_success_params(env):
    return [env._upright, env._standing, env._dont_move]

def get_state_to_save(state):
    return state[:67]

def generate_trajectory(idx, exp_dir, args):
    seed = (os.getpid() * int(time.time())) % 123456789 # To handle random seeds across multiple processes
    args.seed = seed
    trainer = Trainer(args)
    env = trainer.env
    policy = trainer.policy

    state, done = env.reset(test_time=True, speed=args.target_speed), False

    video = []
    initial_state = get_state_to_save(state)
    trajectory = [
        ', '.join([str(i) for i in initial_state])
    ]
    success_params_trajectory = [
        ', '.join(str(i) for i in get_success_params(env))
    ]

    while not done:
        action = policy.select_action(state)

        state, reward, done, _ = env.step(action, test_time=True)
        trajectory.append('\n' + ', '.join([str(i) for i in get_state_to_save(state)]))
        success_params_trajectory.append('\n' + ', '.join([str(i) for i in get_success_params(env)]))

        img = env.render()
        video.append(img)

    if len(video) != 0:
        imageio.mimsave(os.path.join(exp_dir, 'videos/{}.mp4'.format(idx)),video, fps=30)

    with open(os.path.join(exp_dir, 'state/state_trajectory_{}.txt'.format(idx)), 'w+') as file:
        file.writelines(trajectory)

    with open(os.path.join(exp_dir, 'success_params/success_params_trajectory_{}.txt'.format(idx)), 'w+') as file:
        file.writelines(success_params_trajectory)

class Trainer():
    def __init__(self, args):
        args = organize_args(args)
        self.args = args
        if not self.args.get_trajectories:
            self.setup(args)
            self.logger.log_start(sys.argv, args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.env = self.create_env(args)


        obs_dim = self.env.obs_shape
        self.act_dim = self.env.action_space

        self.buf = ReplayBuffer(obs_dim, self.act_dim, args, max_size=int(args.replay_buffer_size))
        self.env.buf = self.buf

        self.policy = SAC(obs_dim, self.act_dim,
                          init_temperature=args.init_temperature,
                          alpha_lr=args.alpha_lr,
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          tau=args.tau,
                          discount=args.discount,
                          critic_target_update_freq=args.critic_target_update_freq,
                          args=args)

        if args.test_policy or args.load_dir:
            self.policy.load(os.path.join(args.load_dir + '/model', 'best_model'), load_optimizer=False)

        if args.teacher_student:
            self.teacher_policy = SAC(self.env.teacher_env.obs_shape,
                                      self.env.teacher_env.action_space,
                                      init_temperature=args.init_temperature,
                                      alpha_lr=args.alpha_lr,
                                      actor_lr=args.actor_lr,
                                      critic_lr=args.critic_lr,
                                      tau=args.tau,
                                      discount=args.discount,
                                      critic_target_update_freq=args.critic_target_update_freq,
                                      args=args)
            self.teacher_policy.load(os.path.join(args.teacher_dir + '/model', 'best_model'), load_optimizer=False)
            for param in self.teacher_policy.parameters():
                param.requires_grad = False
            self.env.set_teacher_policy(self.teacher_policy)

    def setup(self, args):
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H-%M", ts)
        exp_name = args.env + '_' + ts + '_' + 'seed_' + str(args.seed)
        exp_name = exp_name + '_' + args.tag if args.tag != '' else exp_name
        self.experiment_dir = os.path.join(args.work_dir, exp_name)

        utils.make_dir(self.experiment_dir)
        self.video_dir = utils.make_dir(os.path.join(self.experiment_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(self.experiment_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.experiment_dir, 'buffer'))
        self.logger = RLLogger(self.experiment_dir)

        self.save_args(args)

    def save_args(self, args):
        with open(os.path.join(self.experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    def env_function(self):
        if self.args.env == 'HumanoidStandup':
            if self.args.teacher_student:
                return HumanoidStandupVelocityEnv
            return HumanoidStandupEnv
        elif self.args.env == "HumanoidVariantStandup":
            if self.args.teacher_student:
                return HumanoidVariantStandupVelocityEnv
            return HumanoidVariantStandupEnv

    def create_env(self, args):
        env_generator = self.env_function()
        return env_generator(args, args.seed)

    def train_sac(self):
        store_buf = False if self.args.teacher_student else True
        test_time = True if self.args.test_policy else False
        state, done = self.env.reset(store_buf=store_buf, test_time=test_time), False
        t = 0
        self.last_power_update = 0
        self.last_duration = np.inf
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        self.curriculum = True
        best_reward = -np.inf

        while t < int(self.args.max_timesteps):

            # Select action randomly or according to policy
            if self.args.test_policy:
                action = self.policy.select_action(state)
            elif (t < self.args.start_timesteps and not self.args.load_dir):
                action = np.clip(2 * np.random.random_sample(size=self.act_dim) - 1, -self.env.power, self.env.power)
            else:
                action = self.policy.sample_action(state)

            next_state, reward, done, _ = self.env.step(a=action)

            if self.args.test_policy:
                image_l = self.env.render()
                cv2.imshow('image', image_l)
                cv2.waitKey(1)

            episode_timesteps += 1
            self.buf.add(state, action, next_state, reward, self.env.terminal_signal)
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if (t >= self.args.start_timesteps) and not self.args.test_policy:
                self.policy.train(self.buf, self.args.batch_size)

            if done:
                self.logger.log_train_episode(t, episode_num, episode_timesteps, episode_reward, self.policy.loss_dict,
                                              self.env, self.args)
                self.policy.reset_record()
                state, done = self.env.reset(store_buf=store_buf, test_time=test_time), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if t % self.args.test_interval == 0:
                test_reward, min_test_reward, video = self.run_tests(self.env.power_base, self.policy)
                for i, v in enumerate(video):
                    if len(v) != 0:
                        imageio.mimsave(os.path.join(self.video_dir, 't_{}_{}.mp4'.format(t, i)), v, fps=30)
                criteria = test_reward if self.args.avg_reward else min_test_reward
                self.curriculum = self.update_power(self.env, criteria, t)
                if (test_reward > best_reward):
                    self.policy.save(os.path.join(self.model_dir, 'best_model'))
                    best_reward = test_reward
                    self.logger.info("Best model saved")
                self.policy.save(os.path.join(self.model_dir, 'newest_model'))
                self.logger.log_test(test_reward, min_test_reward, self.curriculum, self.env.power_base)
            t += 1

    def update_power(self, env, criteria, t):
        if not self.curriculum:
            return False
        if criteria > self.args.threshold:
            env.power_base = max(env.power_end, 0.95 * env.power_base)
            self.args.curr_power = env.power_base
            self.save_args(self.args)
            if env.power_base == env.power_end:
                return False
            self.last_duration = t - self.last_power_update
            self.last_power_update = t

        else:
            current_stage_length = t - self.last_power_update
            if current_stage_length > min(1000000, max(300000, 1.5 * self.last_duration)) and env.power_base < 1.0:
                env.power_base = env.power_base / 0.95
                self.args.curr_power = env.power_base
                self.save_args(self.args)
                env.power_end = env.power_base
                return False

        return True

    def run_test_iteration(self, test_env, speed_profile, test_policy, video_index, iteration_index, video_array, test_reward):
        video = []
        state, done = test_env.reset(test_time=True, speed=speed_profile[iteration_index]), False
        episode_timesteps = 0
        episode_reward = 0

        while not done:
            action = test_policy.select_action(state)
            state, reward, done, _ = test_env.step(action, test_time=True)
            episode_reward += reward
            episode_timesteps += 1
            if iteration_index in video_index:
                if episode_timesteps == 1:
                    video = video + list(test_env.starting_images)
                video.append(test_env.render())

        if self.args.to_file:
            test_env.geom_traj["state"] = np.stack(test_env.geom_traj["state"])
            test_env.teacher_geoms["state"] = np.stack(test_env.teacher_geoms["state"])
            for name in test_env.geom_names:
                test_env.geom_traj[name + "_pos"] = np.stack(test_env.geom_traj[name + "_pos"])
                test_env.geom_traj[name + "_angleaxis"] = np.stack(test_env.geom_traj[name + "_angleaxis"])
                test_env.teacher_geoms[name + "_pos"] = np.stack(test_env.teacher_geoms[name + "_pos"])
                test_env.teacher_geoms[name + "_angleaxis"] = np.stack(test_env.teacher_geoms[name + "_angleaxis"])
            np.savez(os.path.join(self.buffer_dir, "RecordedMotionSlow{}".format(iteration_index)), **test_env.geom_traj)
            np.savez(os.path.join(self.buffer_dir, "RecordedMotionFast{}".format(iteration_index)), **test_env.teacher_geoms)

        video_array.append(video)
        test_reward.append(episode_reward)

    def run_tests(self, power_base, test_policy):
        video_index = [np.random.random_integers(0, self.args.test_iterations - 1)]
        # video_index = np.arange(self.args.test_iterations)
        np.random.seed(self.args.seed)
        test_env_generator = self.env_function()
        test_env = test_env_generator(self.args, self.args.seed + 10)
        test_env.power = power_base
        if self.args.teacher_student:
            test_env.set_teacher_policy(self.teacher_policy)
        test_reward = []
        speed_profile = np.linspace(self.args.slow_speed, self.args.fast_speed, num=self.args.test_iterations,
                                    endpoint=True)
        video_array = []
        for i in range(self.args.test_iterations):
            self.run_test_iteration(test_env, speed_profile, test_policy, video_index, i, video_array, test_reward)
        test_reward = np.array(test_reward)
        return test_reward.mean(), test_reward.min(), video_array


if __name__ == "__main__":
    main()
