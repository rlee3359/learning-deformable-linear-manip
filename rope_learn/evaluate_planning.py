import multiprocessing as mp
import math
import argparse
import os
from os.path import join, exists
import json
import numpy as np
from tqdm import tqdm
import cv2
import timeit
import copy

import torch
import torch.utils.data as data
from torchvision import transforms

from rope_learn.models import FCNShift
from rope_learn.dataset import TrajectoryDataset
from rope_learn.utils import save_np_images
from PIL import Image

color_dim = None

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_dataloader(params):
    dset = TrajectoryDataset(root=join(params['root'], 'train_data'))
    data_loader = data.DataLoader(dset, batch_size=1, shuffle=True)
    return data_loader

# Assumes input is a [0, 255] numpy array of 64 x 64 x 3
# Processes to [-1, 1] FloatTensor of 3 x 64 x 64
def process_obs(o):
    transform = transforms.Compose([
        transforms.Normalize((0.5,) * color_dim, (0.5,) * color_dim),
    ])
    o = torch.FloatTensor(o / 255.).permute(2, 0, 1).contiguous()
    return transform(o)

# Runs a single element (no batch dimension) through a PyTorch model
def run_single(model, *args):
    return model(*[a.unsqueeze(0) for a in args]).squeeze(0)

def segment_rope(image):
    return np.all(image > 150, axis=2)

def segment_cloth(image):
    image_dim_1 = image[:, :, [1]]
    image_dim_2 = image[:, :, [2]]
    mask = np.all(image > 200, axis=2) + np.all(image_dim_2 < 40, axis=2) + \
           (~np.all(image_dim_1> 135, axis=2))
    mask = mask > 0
    return mask

def segment(image, args):
    if args.mode == 'rope':
        return segment_rope(image)
    elif args.mode == 'cloth':
        return segment_cloth(image)
    else:
        raise Exception('Invalid mode', args.mode)

def sample_pixel(image, args):
    seg = segment(image, args)
    # cv2.imshow('seg',seg*255.0)
    # cv2.waitKey(0)
    idxs = np.argwhere(seg)
    idx = idxs[np.random.randint(len(idxs))]
    idx = np.array([idx[1],idx[0]])
    # print(idx)
    # idx = np.array([32,32])
    # return idx
    return 2 * (idx / 63) - 1

def add_arrow(image, act, args):
    loc = (act[:2] * 0.5 + 0.5) * 63
    loc = loc[[1,0]]
    act = act[2:]
    # act[1] = -act[1]
    act = act[[1, 0]]
    act *= 8#0.3 * 64

    startr, startc = loc
    endr, endc = loc + act
    startr, startc, endr, endc = int(startr), int(startc), int(endr), int(endc)
    cv2.arrowedLine(image, (startc, startr), (endc, endr), (250, 0, 0), 1)
    image[startr-1:startr+1, startc-1:startc+1, :] = (0, 0, 0)

def sample_goal_action(obs, goal, args):
    if args.mode == 'rope':
        loc = sample_pixel(obs, args)
        goal_loc = sample_pixel(goal, args)


        act = ((goal_loc - loc)/8)

        # print(loc,goal_loc,act)

        # act = 2 * act - 1

        loc = 2 * (loc / 63) - 1
        # act = np.array([act[1],act[0]])
        action = np.concatenate((loc, act))
    elif args.mode == 'cloth':
        loc = sample_pixel(obs, args)
        act = 2 * np.random.rand(3) - 1
        action = np.concatenate((loc, act))
    else:
        raise Exception()
    return action

def sample_action(obs, args, env):
    if args.mode == 'rope':
        # loc = sample_pixel(obs, args)
        # act = 2 * np.random.rand(2) - 1
        # action = np.concatenate((loc, act))
        action = env.random_action()
    elif args.mode == 'cloth':
        loc = sample_pixel(obs, args)
        act = 2 * np.random.rand(3) - 1
        action = np.concatenate((loc, act))
    else:
        raise Exception()
    return action


def fwd_plan(env,fwd_model, zcurrent, znext, obs, device, args, n_trials=100, goal_img = None):
    zs, actions = [], []

    zcurrent_b = []
    znext_b = []
    action_b = []
    with torch.no_grad():
        for _ in range(n_trials):
            action = torch.FloatTensor(sample_action(obs, args,env))
            if isinstance(fwd_model, FCNShift) or isinstance(fwd_model, FCNPickPlace):
                zcurrent_i, znext_i, action_i = fwd_model.format_transition(zcurrent, znext, action.to(device))
                zcurrent_b.append(zcurrent_i.cpu())
                znext_b.append(znext_i.cpu())
                action_b.append(action_i.cpu())
            else:
                zcurrent_b.append(zcurrent.cpu())
                znext_b.append(znext.cpu())
                action_b.append(action)
            actions.append(action)


        torch.cuda.empty_cache()
        zcurrent_b = torch.stack(zcurrent_b)
        znext_b = torch.stack(znext_b)
        action_b = torch.stack(action_b)

        zs = fwd_model(zcurrent_b.to(device), action_b.to(device))
        znext_b = znext_b.to(device)

        dists = torch.norm((zs - znext_b).contiguous().view(n_trials, -1), dim=-1)
        idx = torch.argmin(dists)
        torch.cuda.empty_cache()

        return actions[idx], zs[idx]


def compute_single_trial(encoder, inv_model, fwd_model_linear, fwd_model_mlp, trans, data_loader,
                             env, o_flat, state_flat, n_actions, device, args):
    obs, actions, env_states = next(iter(data_loader))
    obs = obs.squeeze(0).to(device)
    actions, env_states = actions.squeeze(0), env_states.squeeze(0)
    traj_length = obs.shape[0]
    start_t = np.random.randint(traj_length)

    actions, env_states = actions.numpy(), env_states.numpy()
    # print(env_states[start_t])
    env.set_state(env_states[start_t])
    # original = env.get_obs().pixels.astype('uint8')
    
    original = env.get_image().astype('uint8')
    trajectory = [original]

    zstart = run_single(encoder, process_obs(original).cuda())
    if args.goal_type == 'random':
        obs_other, _, env_states_other = next(iter(data_loader))
        obs_other = obs_other.squeeze(0).to(device)
        env_states_other = env_states_other.squeeze(0).numpy()
        end_t = np.random.randint(obs_other.shape[0])
        zend = run_single(encoder, obs_other[end_t])
    elif args.goal_type == 'flat':
        zend = run_single(encoder, process_obs(o_flat).to(device))
    elif args.goal_type == 'rope45':
        o_45 = np.array(Image.open('imgs/rope_45.png'))
        zend = run_single(encoder, process_obs(o_45).to(device))
    elif args.goal_type == 'rope90':
        o_90 = cv2.rotate(o_flat, cv2.ROTATE_90_CLOCKWISE)
        # o_90 = np.array(Image.open('imgs/rope_90.png'))
        zend = run_single(encoder, process_obs(o_90).to(device))
    elif args.goal_type == 'rope135':
        o_135 = np.array(Image.open('imgs/rope_135.png'))
        zend = run_single(encoder, process_obs(o_135).to(device))
    else:
        raise Exception('Invalid goal_type', args.goal_type)

    for t in range(n_actions):
        if len(np.argwhere(segment(trajectory[-1], args))) == 0:
            break

        # Plan using one-step MPC for the next action
        if args.action_type == 'trans':
            start_time = timeit.default_timer()
            action, z_chosen = fwd_plan(env,trans, zstart, zend, trajectory[-1], device, args)
            # print("Time To Plan: {}".format(timeit.default_timer() - start_time))
        elif args.action_type == 'random':
            action = env.action_space.sample()
            action[:2] = sample_pixel(trajectory[-1], args)
            action = torch.FloatTensor(action)
        elif args.action_type == 'none':
            action = torch.zeros(env.action_space.shape)
        else:
            raise Exception('Invalid action_type', args.action_type)

        # Add arrows for action visualization
        trajectory[-1] = trajectory[-1].astype('uint8')
        add_arrow(trajectory[-1], action.cpu().numpy().copy(), args)
        trajectory[-1] = trajectory[-1].astype('float32')

        # Step through the environment
        # o = env.step(action.cpu().numpy())[0].pixels
        o = env.step(action.cpu().numpy())[0]
        cv2.imshow("o",o)
        cv2.waitKey(1)
        trajectory.append(o)
        zstart = run_single(encoder, process_obs(o).to(device))
    if len(trajectory) < n_actions+1:
        trajectory.extend([trajectory[-1] for _ in range(n_actions+1 - len(trajectory))])

    plan_goal = o.astype('uint8')
    if args.goal_type == 'flat':
        true_goal = o_flat
    elif args.goal_type == 'rope45':
        true_goal = o_45
    elif args.goal_type == 'rope90':
        true_goal = o_90
    elif args.goal_type == 'rope135':
        true_goal = o_135
    else:
        true_goal = (obs_other[end_t].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    trajectory.append(true_goal.astype('float32'))
    trajectory = np.stack(trajectory, axis=0)

    plan_geoms = env.get_state().copy()
    # plan_geoms = env.get_geoms().copy()

    if args.goal_type == 'flat' or args.goal_type == 'random':
        if args.goal_type == 'flat':
            env.set_state(state_flat)
        else:
            env.set_state(env_states_other[end_t])
        true_geoms = env.get_state().copy()
        # true_geoms = env.get_geoms().copy()
        if args.mode == 'rope':
            # print(plan_geoms)
            # print(true_geoms)
            plan_geoms = np.array([p[0] for p in plan_geoms])
            true_geoms = np.array([p[0] for p in true_geoms])
            geom_error = min(np.linalg.norm(plan_geoms - true_geoms, axis=-1).sum(),
                             np.linalg.norm(plan_geoms - true_geoms[::-1], axis=-1).sum())
        elif args.mode == 'cloth':
            r1_true_geoms = rotate_tensor(true_geoms)
            r2_true_geoms = rotate_tensor(r1_true_geoms)
            r3_true_geoms = rotate_tensor(r2_true_geoms)
            geom_error = min(np.linalg.norm(plan_geoms - true_geoms, axis=-1).sum(),
                             np.linalg.norm(plan_geoms - r1_true_geoms, axis=-1).sum(),
                             np.linalg.norm(plan_geoms - r2_true_geoms, axis=-1).sum(),
                             np.linalg.norm(plan_geoms - r3_true_geoms, axis=-1).sum())
            geom_error = float(geom_error)
        else:
            raise Exception()
    else:
        if args.goal_type == 'rope45':
            true_geoms = np.load('imgs/rope_45.npy').astype('float32')
        elif args.goal_type == 'rope90':
            true_geoms = np.load('imgs/rope_90.npy').astype('float32')
        elif args.goal_type == 'rope135':
            true_geoms = np.load('imgs/rope_135.npy').astype('float32')
        geom_error = min(np.linalg.norm(plan_geoms - true_geoms, axis=-1).sum(),
                         np.linalg.norm(plan_geoms - true_geoms[::-1], axis=-1).sum())

    plan_goal_mask, true_goal_mask = segment(plan_goal, args), segment(true_goal, args)
    xor = np.logical_xor(plan_goal_mask, true_goal_mask).sum()
    iou = np.logical_and(plan_goal_mask, true_goal_mask).sum() / np.logical_or(plan_goal_mask, true_goal_mask).sum()
    return dict(xor=xor, iou=iou, geom=geom_error), original, plan_goal, true_goal, trajectory


def rotate_tensor(tensor):
    assert tensor.shape[0] == tensor.shape[1], tensor.shape
    N = tensor.shape[0]
    tensor = tensor.copy()
    for x in range(0, int(N / 2)):
        for y in range(x, N - x - 1):
            temp = tensor[x, y].copy()
            tensor[x, y] = tensor[y, N - 1 - x].copy()
            tensor[y, N - 1 - x] = tensor[N - 1 - x, N - 1 - y].copy()
            tensor[N - 1 - x, N - 1 - y] = tensor[N - 1 - y, x].copy()
            tensor[N - 1 - y, x] = temp
    return tensor



def compute_average_error(encoder, inv_model, fwd_model_linear, fwd_model_mlp, trans,
                          data_loader, env_args, n_actions, device, args):

    from envs.real_env import RopeEnv
    env = RopeEnv()
    o_flat = env.reset(randomize_start=False)#.pixels
    state_flat = env.get_state()

    results = dict()
    start_imgs, plan_imgs, true_imgs, trajectories = [], [], [], []

    pbar = tqdm(total=args.n_trials)
    for t in range(args.n_trials):
        with torch.no_grad():
            out, start, plan, true, traj = compute_single_trial(encoder, inv_model, fwd_model_linear, fwd_model_mlp, trans, data_loader,
                                                                env, o_flat, state_flat, n_actions, device, args)
            if t < 33: # Only keep 33 of them so ~100 images total
                start_imgs.append(start)
                plan_imgs.append(plan)
                true_imgs.append(true)
                trajectories.append(traj)

        for key, val in out.items():
            if key not in results:
                results[key] = []
            results[key].append(val)
        pbar.update(1)
    pbar.close()

    start_imgs = np.stack(start_imgs, axis=0)
    plan_imgs = np.stack(plan_imgs, axis=0)
    true_imgs = np.stack(true_imgs, axis=0)
    trajectories = np.concatenate(trajectories, axis=0)
    return {key: np.mean(val) for key, val in results.items()}, start_imgs, plan_imgs, true_imgs, trajectories

def main(args):
    args, id = args
    args = AttrDict(args)
    np.random.seed(args.seed + id)
    torch.manual_seed(args.seed + id)
    torch.cuda.manual_seed(args.seed + id)

    assert exists(args.folder)

    with open(join(args.folder, 'params.json')) as f:
        params = json.load(f)

    with open(join(params['root'], 'env_args.json'), 'r') as f:
        env_args = json.load(f)
    env_args['max_path_length'] = 1200
    if 'task_kwargs' not in env_args:
        env_args['task_kwargs'] = {}
    env_args['task_kwargs']['init_flat'] = True
    env_args['task_kwargs']['random_pick'] = False

    device = torch.device('cuda:0')
    checkpoint = torch.load(join(args.folder, 'checkpoint'), map_location=device)
    encoder = checkpoint['encoder']
    inv_model, fwd_model_linear, fwd_model_mlp, trans = None, None, None, None
    if args.action_type == 'inv_model':
        inv_model = torch.load(join(args.folder, 'inv_model.pt'), map_location=device)
    elif args.action_type == 'trans':
        trans = checkpoint['trans']
    elif args.action_type == 'fwd_model_linear':
        fwd_model_linear = torch.load(join(args.folder, 'fwd_model_linear.pt'), map_location=device)
    elif args.action_type == 'fwd_model_mlp':
        fwd_model_mlp = torch.load(join(args.folder, 'fwd_model_mlp.pt'), map_location=device)
    elif args.action_type != 'random' and args.action_type != 'none':
        raise Exception(args.action_type)

    global color_dim
    color_dim = 3

    data_loader = get_dataloader(params)
    res, start_imgs, plan_imgs, true_imgs, trajectories = compute_average_error(encoder, inv_model,
                                                                                fwd_model_linear, fwd_model_mlp, trans,
                                                                                data_loader, env_args,
                                                                                args.n_actions, device, args)

    if id == 0:
        prefix = f'[action_type]_{args.action_type}_[goal_type]_{args.goal_type}'
        eval_folder = join(args.folder, 'eval', prefix)
        if not exists(eval_folder):
            os.makedirs(eval_folder)

        imgs = np.stack((start_imgs, plan_imgs, true_imgs), axis=1).astype('float32')
        imgs = imgs.reshape(-1, *imgs.shape[-3:])
        save_np_images(imgs, join(eval_folder, f'eval_{prefix}.png'), nrow=12)
        save_np_images(trajectories, join(eval_folder, f'traj_{prefix}.png'), nrow=args.n_actions + 2)

    save_args = vars(args)
    save_args['script'] = 'evaluate_planning'
    res.update(save_args)
    res.update(params)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--mode', type=str, default='rope', help='rope|cloth')
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--action_type', type=str, default='trans', help='trans|random|none')
    parser.add_argument('--goal_type', type=str, default='random', help='random|flat|rope45|rope90|rope135')
    parser.add_argument('--n_actions', type=int, default=20)
    parser.add_argument('--n_cpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.n_cpu = 5
    if args.n_cpu == -1:
        args.n_cpu = mp.cpu_count() // 4
    args.n_trials = math.ceil(args.n_trials // args.n_cpu)

    args_dict = vars(args)
    pool = mp.Pool(processes=args.n_cpu)
    rtn_dicts = pool.map(main, list(zip([args_dict] * args.n_cpu, range(args.n_cpu))))

    results = rtn_dicts[0]
    keys = ['xor', 'iou', 'geom']
    for k in keys:
        vals = [r[k] for r in rtn_dicts]
        v = float(np.mean(vals))
        results[k] = v

    prefix = f'[action_type]_{args.action_type}_[goal_type]_{args.goal_type}'
    eval_folder = join(args.folder, 'eval', prefix)
    if not exists(eval_folder):
        os.makedirs(eval_folder)

    with open(join(eval_folder, 'eval_results.json'), 'w') as f:
        json.dump(results, f)
    print(results)

