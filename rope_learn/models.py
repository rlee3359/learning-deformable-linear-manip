import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim):
        super().__init__()

        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim, trans_type='linear'):
        super().__init__()
        if trans_type in ['linear', 'mlp']:
            self.model = TransitionSimple(z_dim, action_dim, trans_type=trans_type)
        elif 'reparam_w' in trans_type:
            self.model = TransitionParam(z_dim, action_dim, hidden_sizes=[64, 64],
                                         orthogonalize_mode=trans_type)
        else:
            raise Exception('Invalid trans_type:', trans_type)

    def forward(self, z, a):
        return self.model(z, a)


class TransitionSimple(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, trans_type='linear'):
        super().__init__()
        self.trans_type = trans_type
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 64
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x


class TransitionParam(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, hidden_sizes=[], orthogonalize_mode='reparam_w'):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.orthogonalize_mode = orthogonalize_mode

        if orthogonalize_mode == 'reparam_w_ortho_cont':
            self.model = MLP(z_dim + action_dim, z_dim * (z_dim - 1), hidden_sizes=hidden_sizes)
        else:
            self.model = MLP(z_dim + action_dim, z_dim * z_dim, hidden_sizes=hidden_sizes)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        if self.orthogonalize_mode == 'reparam_w':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
        elif self.orthogonalize_mode == 'reparam_w_ortho_gs':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
            Ws = orthogonalize_gs(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_ortho_cont':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim - 1)  # b x z_dim x z_dim - 1
            Ws = orthogonalize_cont(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_tanh':
            Ws = torch.tanh(self.model(x)).view(x.shape[0], self.z_dim, self.z_dim) / math.sqrt(self.z_dim)
        else:
            raise Exception('Invalid orthogonalize_mode:', self.orthogonalize_mode)
        return torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1) # b x z_dim


# Gram-Schmidt
def orthogonalize_gs(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)
    return Ws_new


def orthogonalize_cont(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim - 1):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)

    # Ws_new is b x z_dim x z_dim - 1
    determinants = []
    for k in range(z_dim):
        tmp = torch.cat((Ws_new[:, :k], Ws_new[:, k+1:]), dim=1).permute(0, 2, 1).contiguous()
        tmp = tmp.cpu()
        det = torch.det(tmp)
        det = det.cuda()
        if k % 2 == 1:
            det = det * -1
        determinants.append(det)
    determinants = torch.stack(determinants, dim=-1).unsqueeze(-1) # b x z_dim x 1
    determinants = determinants / torch.norm(determinants, dim=1, keepdim=True)
    Ws_new = torch.cat((Ws_new, determinants), dim=-1)
    return Ws_new


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop() # Pop last ReLU
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= n_bit ** 2 - 1 # [0, 15] for n_bit = 4
    x = torch.floor(x + 1e-4) # [0, 15]
    return x

class Decoder(nn.Module):
    prefix = 'decoder'

    def __init__(self, z_dim, channel_dim, discrete=False, n_bit=4):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        self.discrete = discrete
        self.n_bit = n_bit
        self.discrete_dim = 2 ** n_bit

        out_dim = self.discrete_dim * self.channel_dim if discrete else channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)

        if self.discrete:
            output = output.view(output.shape[0], self.discrete_dim,
                                 self.channel_dim, *output.shape[2:])
        else:
            output = torch.tanh(output)

        return output


    def loss(self, x, z):
        recon = self(z)
        if self.discrete:
            loss = F.cross_entropy(recon, quantize(x, self.n_bit).long())
        else:
            loss = F.mse_loss(recon, x)
        return loss


    def predict(self, z):
        recon = self(z)
        if self.discrete:
            recon = torch.max(recon, dim=1)[1].float()
            recon = (recon / (self.discrete_dim - 1) - 0.5) / 0.5
        return recon


class InverseModel(nn.Module):
    prefix = 'inv'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)


class ForwardModel(nn.Module):
    prefix = 'forward'

    def __init__(self, z_dim, action_dim, mode='linear'):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        if mode == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        else:
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, z_dim),
            )

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        return self.model(x)


class PixelForwardModel(nn.Module):
    def __init__(self, obs_dim, action_dim, learn_delta=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learn_delta = learn_delta

        self.conv1 = nn.Conv2d(obs_dim[0], 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv6 = nn.Conv2d(256, 256, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

        self.fc1 = nn.Linear(action_dim, 256)
        self.fc2 = nn.Linear(action_dim, 128)
        self.fc3 = nn.Linear(action_dim, 64)
        self.fc4 = nn.Linear(action_dim, 64)

    def forward(self, obs, actions):
        out = F.leaky_relu(self.conv1(obs), 0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), 0.2, inplace=True)
        h1 = F.leaky_relu(self.conv3(out),  0.2, inplace=True) # 32 x 32
        h2 = F.leaky_relu(self.conv4(h1),   0.2, inplace=True) # 16 x 16
        h3 = F.leaky_relu(self.conv5(h2),   0.2, inplace=True) # 8 x 8
        out = F.leaky_relu(self.conv6(h3),  0.2, inplace=True) # 4 x 4

        out = F.leaky_relu(self.deconv1(out) * self.fc1(actions).unsqueeze(-1).unsqueeze(-1), 0.2, inplace=True) + h3
        out = F.leaky_relu(self.deconv2(out) * self.fc2(actions).unsqueeze(-1).unsqueeze(-1), 0.2, inplace=True) + h2
        out = F.leaky_relu(self.deconv3(out) * self.fc3(actions).unsqueeze(-1).unsqueeze(-1), 0.2, inplace=True) + h1
        out = F.leaky_relu(self.deconv4(out) * self.fc4(actions).unsqueeze(-1).unsqueeze(-1), inplace=True)
        out = self.out_conv(out)

        if self.learn_delta:
            out = obs + out
        return out



class FCNShift(nn.Module):
    def __init__(self, obs_dim, action_dim, learn_delta=False, rotate=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.W = obs_dim[1]*2
        self.action_dim = action_dim
        self.learn_delta = learn_delta

        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv6 = nn.Conv2d(256, 256, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.out_conv = nn.Conv2d(64, 1, 3, 1, 1)


    def forward(self, obs, act):
        x = torch.cat([obs,act],dim=1)
        out = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), 0.2, inplace=True)
        h1 = F.leaky_relu(self.conv3(out),  0.2, inplace=True) # 32 x 32
        h2 = F.leaky_relu(self.conv4(h1),   0.2, inplace=True) # 16 x 16
        h3 = F.leaky_relu(self.conv5(h2),   0.2, inplace=True) # 8 x 8
        out = F.leaky_relu(self.conv6(h3),  0.2, inplace=True) # 4 x 4

        out = F.leaky_relu(self.deconv1(out), 0.2, inplace=True) + h3
        out = F.leaky_relu(self.deconv2(out), 0.2, inplace=True) + h2
        out = F.leaky_relu(self.deconv3(out), 0.2, inplace=True) + h1
        out = F.leaky_relu(self.deconv4(out), inplace=True)
        out = F.upsample_bilinear(out, size=(self.W,self.W))
        out = self.out_conv(out)

        if self.learn_delta:
            out = obs + out
        return out


    def viz(self, obs, obs_pos, actions, obs_pos_h, actions_vec):
        obs = obs[0]
        obs_pos = obs_pos[0]
        actions = actions[0]
        act = actions_vec[0]

        obs_pos_h = obs_pos_h[0]

        obs = obs.permute(1,2,0).cpu().numpy()
        obsn = obs_pos.permute(1,2,0).cpu().numpy()
        obsnh = obs_pos_h.permute(1,2,0).cpu().numpy()
        obs = (obs / 2) + 0.5
        obsn = (obsn / 2) + 0.5
        obsnh = (obsnh / 2) + 0.5

        delta_pixels = 8
        mean = torch.FloatTensor([0.5,0.5,0.0,0.0]).cuda()
        std = torch.FloatTensor([0.5,0.5,1.0,1.0]).cuda()
        # Un-normalize
        act = act*std + mean
        act = act.cpu().numpy()
        pick = [64,64]
        pick = int(pick[1]), int(pick[0])
        place = int(pick[0] + act[3]*delta_pixels), int(pick[1]+act[2]*delta_pixels)

        obsi = (obs / 2) + 0.5
        obsni = (obsn / 2) + 0.5
        obs_img_arrow = np.ascontiguousarray(0.5*obsi + 0.5*obsnh)
        cv2.arrowedLine(obs_img_arrow,(pick[1],pick[0]), (place[1],place[0]), (0,0,200),1)
        img = np.hstack([obs, obsn, obsnh, obs_img_arrow])

        return img

    def format_transition(self, obs, nobs, act):
        delta_movement = act[2:].cpu().numpy()
        dist = np.linalg.norm(delta_movement)/np.linalg.norm([1,1])
        mean = torch.FloatTensor([0.5,0.5,0.0,0.0]).cuda()
        std = torch.FloatTensor([0.5,0.5,1.0,1.0]).cuda()
        # Un-normalize
        act = act*std + mean

        IM_W = 64
        delta_pixels = 8
        PAD_W = 90

        obs = obs.permute(1,2,0).cpu().numpy()
        nobs = nobs.permute(1,2,0).cpu().numpy()
        act = act.cpu().numpy()
        place_channel = np.zeros((64, 64))
        pick_channel = np.zeros((64, 64))

        pick = act[:2]*64
        pick = int(pick[1]), int(pick[0])
        place = int(pick[0] + act[3]*delta_pixels), int(pick[1]+act[2]*delta_pixels)

        obsi = (obs / 2) + 0.5
        nobsi = (nobs / 2) + 0.5
        obs_img_arrow = np.ascontiguousarray(0.5*obsi + 0.5*nobsi)

	
        cv2.arrowedLine(obs_img_arrow,(pick[1],pick[0]), (place[1],place[0]), (0,0,200),1)

        pick_channel[pick[0],pick[1]] = 1.0
        place_channel[place[0],place[1]] = 1.0

        obs_pad = np.zeros((self.W, self.W, 1))
        obs_pad[64-pick[0]:64-pick[0]+64, 64-pick[1]:64-pick[1]+64] = obs
        
        nobs_pad = np.zeros((self.W, self.W, 1))
        nobs_pad[64-pick[0]:64-pick[0]+64, 64-pick[1]:64-pick[1]+64] = nobs
        place_pad = np.zeros((self.W, self.W))
        place_pad[64-pick[0]:64-pick[0]+64, 64-pick[1]:64-pick[1]+64] = place_channel

        angle = np.degrees(np.arctan2(act[3]*delta_pixels, act[2]*delta_pixels))
        # print(angle)
        obs_pad = self.rotate(obs_pad, angle)
        nobs_pad = self.rotate(nobs_pad, angle)
        place_pad = self.rotate(place_pad, angle)
        delta_img = np.zeros_like(place_pad)
        delta_img[:] = dist

        img_pad = np.vstack([obs_pad, nobs_pad])#, place_pad])
        # cv2.imshow("img_pad", img_pad)
        # cv2.imshow("place_pad", place_pad)
        # cv2.imshow("delta_img", delta_img)

        # cv2.waitKey(0)
        obs   = torch.FloatTensor(obs_pad).unsqueeze(2).cuda().permute(2,0,1)
        nobs  = torch.FloatTensor(nobs_pad).unsqueeze(2).cuda().permute(2,0,1)
        act  = torch.FloatTensor(delta_img).cuda().unsqueeze(0)
        # act  = torch.FloatTensor(act).cuda()
        # place = torch.FloatTensor(place_shift).cuda().unsqueeze(0)
        return obs, nobs, act


    def rotate(self, image, angle, scale=1.0):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # return the rotated image
        return rotated


    def format_batch(self, obs, obs_pos, actions):
        obs_f, obs_pos_f, actions_f = [], [], []
        for o, o_p, a in zip(obs, obs_pos, actions):
            o, o_p, a = self.format_transition(o, o_p, a)
            obs_f.append(o)
            obs_pos_f.append(o_p)
            actions_f.append(a)
        obs_f = torch.stack(obs_f)
        obs_pos_f = torch.stack(obs_pos_f)
        actions_f = torch.stack(actions_f)
        return obs_f, obs_pos_f, actions_f

