import pybullet as p
import numpy as np
import cv2
import copy
import time
import random
import torch
# from params import *

#========================================================================================
# Simulated Rope Manipulation Environment
#========================================================================================

class SimRopeEnv:
    def __init__(self, traj_length=10):
        # Create the physics client
        physicsClient = p.connect(p.DIRECT)

        # Time Limit
        self.TL = traj_length

        # Image observation size
        self.W = self.H = 64
        # Cam dist from origin
        cam_dist = -0.4
        # Field of view
        fov = 60
        # Calculate the world space in the image
        self.view_span = (abs(cam_dist)*np.tan(np.deg2rad(30)))#*0.9

        # Actionspace is pick location, movement vector (x,y)
        self.act_size = 4
        self.delta_pixels = 8

        # Create the camera params
        self.view = p.computeViewMatrixFromYawPitchRoll([0,0,0],cam_dist,0,90,0,2)
        self.proj = p.computeProjectionMatrixFOV(fov=fov,aspect=self.W/self.H, nearVal=0.01, farVal=11)

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0,0,-10)
        planeId = p.loadURDF("models/plane.urdf")
        p.changeVisualShape(planeId, -1, rgbaColor=(0,0.2,0.1,1))

        # self.create_walls()

        self.create_rope()
        self.start_state = self.get_state()

        # self.rope_texture = p.loadTexture("models/whiterope.png")
        # self.table_texture = p.loadTexture("models/table.png")

    def create_walls(self):
        wall_height = 0.1
        wv = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01,self.view_span, wall_height])
        wc = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01,self.view_span, wall_height])
        b = p.createMultiBody(0,wc, wv, basePosition=[self.view_span,0,0.01])
        p.changeVisualShape(b, -1, rgbaColor=[0,0,0,0])

        wv = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01,self.view_span,wall_height])
        wc = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01,self.view_span, wall_height])
        b = p.createMultiBody(0,wc, wv, basePosition=[-self.view_span,0,0.01])
        p.changeVisualShape(b, -1, rgbaColor=[0,0,0,0])

        wv = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.view_span,0.01,wall_height])
        wc = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.view_span,0.01, wall_height])
        b = p.createMultiBody(0,wc, wv, basePosition=[0,self.view_span,0.01])
        p.changeVisualShape(b, -1, rgbaColor=[0,0,0,0])

        wv = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.view_span,0.01,wall_height])
        wc = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.view_span,0.01, wall_height])
        b = p.createMultiBody(0,wc, wv, basePosition=[0,-self.view_span,0.01])
        p.changeVisualShape(b, -1, rgbaColor=[0,0,0,0])

    #------------------------------------------------------------------------------------

    def create_rope(self):
        length    = 0.30
        num_links = 25
        width     = 0.015
        self.link_radius = width/2
        cwidth    = 0.016
        link_mass = 0.1

        distance = length/num_links
        link_length = length/(num_links*1.5)
        link_width = width
        pos = [-length/2 - distance/2, 0, link_width/2]

        part_shape  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[link_length/2, cwidth/2, link_width/2])
        part_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=link_width/2, length=distance, visualFrameOrientation=(0.707,0,0.707,0))

        self.link_ids = []
        for l in range(num_links):
            pos[0] += distance
            link_id = p.createMultiBody(link_mass, part_shape, part_visual, basePosition=pos)
            if len(self.link_ids) > 0:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.link_ids[-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=link_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0,0,0),
                    parentFramePosition=(distance/2,0,0),
                    childFramePosition=(-distance/2,0,0))
                p.changeConstraint(constraint_id,maxForce=100)
            # color = [0.6,0,0,1]
            p.changeVisualShape(link_id, -1, rgbaColor=[0.9,0.9,0.9,1]) #textureUniqueId=self.rope_texture)#rgbaColor=color)
            self.link_ids.append(link_id)

    #------------------------------------------------------------------------------------
    
    def get_state(self):
        # Get pos and angles
        link_states = [p.getBasePositionAndOrientation(i) for i in self.link_ids]
        link_states = [list(s[0]) + list(s[1]) for s in link_states]
        link_positions = np.array(link_states)
        return link_positions

    #------------------------------------------------------------------------------------

    def get_geoms(self):
        # Just pos
        link_positions = np.array([list(p.getBasePositionAndOrientation(i)[0]) for i in self.link_ids])
        return link_positions

    #------------------------------------------------------------------------------------

    def set_state(self, state):
        for i,s in enumerate(state):
            s = s[:3], s[3:]
            # print(s)
            p.resetBasePositionAndOrientation(i+1,*s)
        
    #------------------------------------------------------------------------------------

    def randomize_start(self):
        # Apply a number of random actions to randomize the initial conditions
        num_random_acts = 10
        for i in range(num_random_acts):
            act = self.random_action()
            # # print(act)
            # act = act[0], act[1], act[2]*3, act[3]*3
            self.perform_action(act)

    #------------------------------------------------------------------------------------

    def reset(self, rope_params=None, test=False, randomize_rope=False, randomize_start=True):
        self.ts = 0
        self.set_state(self.start_state)

        self.goal = self._get_obs()

        # Apply random actions to randomly initialize rope position
        if randomize_start:
            # print("Randomizing Start")
            self.randomize_start()

        self.step_sim_for_time(0.1)

        self.gripperId = None
        self.goal_pos = [1,1,0.5]
        return self._get_obs()

    #------------------------------------------------------------------------------------

    def perform_action(self, action):
        # x,y is -1,1 so scale to meters
        x =  action[0]*self.view_span
        y =- action[1]*self.view_span
        # end is 
        del_x = ((action[2]*self.delta_pixels)/self.W)*2*self.view_span
        del_y = -((action[3]*self.delta_pixels)/self.W)*2*self.view_span
        end_x = x + del_x
        end_y = y + del_y

        start = [x,     y,     self.link_radius]
        end   = [end_x ,end_y, self.link_radius*3]

        p.addUserDebugLine(start, end, [0.1,0.8,0.3], 5)

        self.pick_and_place(action, start, end)

    #------------------------------------------------------------------------------------

    def step(self, action):
        self.ts += 1
        self.perform_action(action)
        return self._get_obs(), self._get_rew(), self._done(), {}

    #------------------------------------------------------------------------------------

    def act_to_pix(self,action):
        st = int(((action[0]+1)/2)*self.W),int(((action[1]+1)/2)*self.W)
        en = int(st[0] + action[2]*self.delta_pixels), int(st[1]+action[3]*self.delta_pixels)
        return st, en


    def draw_action(self, action, start_angle):
        img = self.get_image()
        st, en = self.act_to_pix(action)
        img = cv2.arrowedLine(img, st, en, (0,100,200), 1)
        # cv2.imshow("step_viz", img)
        # cv2.waitKey(0)

    #------------------------------------------------------------------------------------

    def pick_and_place(self, action, start, end, angle_delta=0):
        # Get positions of links in rope, get distances of these to the requested grasp location
        link_positions = np.array([list(p.getBasePositionAndOrientation(i)[0]) for i in self.link_ids])
        grasp_xy = np.array([start[0],start[1]])
        xys = link_positions[:,:2]
        dist_to_start = np.linalg.norm((xys-grasp_xy),axis=1)

        # Sort the distances, get the link with the smallest dist to grasp pos
        dists_sorted = np.sort(dist_to_start)
        inds_sorted  = np.argsort(dist_to_start)
        grasp_link = inds_sorted[0]

        grasp_pos = np.array(link_positions[grasp_link])

        start_angle = self.create_gripper(grasp_pos, grasp_link+1) # grasp_link is not the pybullet ind, add 1
        self.draw_action(action, start_angle)

        self.move_gripper_to_goal(end, angle_delta)

        if self.gripperId:
          p.removeBody(self.gripperId)
          self.gripperId = None

        self.step_sim_for_time(1)

    #------------------------------------------------------------------------------------

    def step_sim_for_time(self, t):
        step = 0.01
        elapsed = 0
        while elapsed < t:
            p.stepSimulation()
            elapsed += step
            # time.sleep(0.001)

    #------------------------------------------------------------------------------------

    def create_gripper(self,pos, grasp_link):
        # Create the gripper at the link position, join
        part_shape  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.link_radius]*3)
        part_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.link_radius]*3)
        orien = p.getBasePositionAndOrientation(grasp_link)[1]
        orien_e = p.getEulerFromQuaternion(orien)
        new_orien_e = [0,0,orien_e[2]]
        orien = p.getQuaternionFromEuler(new_orien_e)

        self.gripperId = p.createMultiBody(0.0, part_shape, part_visual, basePosition=pos, baseOrientation=orien)
        p.setCollisionFilterGroupMask(self.gripperId,-1,0,0)
        constraint_id = p.createConstraint(grasp_link,-1, self.gripperId, -1, p.JOINT_FIXED, [0,0,0], [0,0,0],[0,0,0])
        return orien_e[2]

    #------------------------------------------------------------------------------------

    def move_gripper_to_goal(self,goal,angle_delta):
        pos_goal = np.array(goal)

        start_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.gripperId)[1])[2]
        ori_goal = start_ori + angle_delta

        def pos_error():
            pos = np.array(p.getBasePositionAndOrientation(self.gripperId)[0])
            return pos_goal-pos

        def wrap_to_pi(angle):
            return np.arctan2(np.sin(angle), np.cos(angle))

        def ori_error():
            ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.gripperId)[1])[2]
            return wrap_to_pi(ori_goal - ori)

        def not_at_goal():
            not_pos_reached = np.linalg.norm(pos_error()) > 0.01
            not_ori_reached = np.linalg.norm(ori_error()) > 0.01
            return not_pos_reached or not_ori_reached

        while not_at_goal():
            e = pos_error()
            v = 0.1*(e)/np.linalg.norm(e)
            ori_e = ori_error()
            twist_vel = 0.1*(ori_e/np.linalg.norm(e))
            av = [0,0,twist_vel]
            p.resetBaseVelocity(self.gripperId,v,av)
            p.stepSimulation()

    #------------------------------------------------------------------------------------

    def get_image(self):
      img = p.getCameraImage(self.W,self.H,self.view,self.proj,renderer=p.ER_BULLET_HARDWARE_OPENGL,shadow=0)[2]
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      img = cv2.flip(img, -1)
      return img

    #------------------------------------------------------------------------------------

    def get_mask(self, img):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = np.zeros_like(grey)
        mask[grey > 100] = 255
        # kernel = np.ones((3,3),np.uint8)
        # mask = cv2.dilate(mask, kernel, 1)
        # mask = cv2.erode(mask, kernel, 1)
        return mask

    #------------------------------------------------------------------------------------

    def _get_obs(self):
      img = self.get_image()
      return img#, self.get_mask(img)

    #------------------------------------------------------------------------------------

    def _get_rew(self):
      cube_pos = p.getBasePositionAndOrientation(self.link_ids[0])[0]
      dist = np.array(cube_pos) - np.array(self.goal_pos)
      return -abs(dist)
   
    #------------------------------------------------------------------------------------

    def _done(self):
      return self.ts >= self.TL# or self.get_mask(self.get_image()).sum() == 0

    #------------------------------------------------------------------------------------

    def random_action(self):
      invalid = True
      while invalid:
          mask = self.get_mask(self.get_image())
          rope_idx = np.transpose(np.nonzero(mask))
          pick  = rope_idx[np.random.randint(rope_idx.shape[0])]
          pick = 2*(pick[0]/self.W)-1, 2*(pick[1]/self.W)-1
          delta = np.random.uniform(-1.0,1.0,(2,))
          act =[pick[1],pick[0],delta[1],delta[0]]

          st, en = self.act_to_pix(act)
          invalid = en[0] > 63 or en[0] < 0 or en[1] > 63 or en[1] < 0
      return act

#========================================================================================
# Helpers
#========================================================================================

down = False
ix,iy = -1, -1
tipx,tipy = -1, -1
def mouse_callback(event, x,y,flags,param):
    global down, img, ix,iy,tipx,tipy
    if event == cv2.EVENT_LBUTTONDOWN:
        down = True
        ix,iy = x,y
        tipx,tipy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if down:
            tipx,tipy = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        down = False

#----------------------------------------------------------------------------------------

def get_user_action(obs_img):
  cv2.namedWindow('Demo')
  cv2.setMouseCallback('Demo', mouse_callback)
  while not down:
    img = copy.deepcopy(obs_img)
    cv2.imshow("Demo", img)
    cv2.waitKey(1)

  while down:
    img = copy.deepcopy(obs_img)
    ovr = np.zeros_like(img)

    xl = tipx - ix
    yl = -(tipy - iy)
    r  = int(np.sqrt(xl**2 + yl**2))
    angle = np.arctan2(yl,xl)

    x = (ix/env.W)
    y = ((iy/env.W))
    xd = (xl/env.W)
    yd = (yl/env.W)
    dist = r/(np.sqrt(env.W**2 + env.H**2))
    act = [x, y, xd, yd]

    cv2.circle(img, (ix,iy),4,(0,100,0),1)
    img = img/255.0 + 0.2*ovr/255.0
    ex = int(ix + r*np.cos(angle))
    ey = int(iy + -r*np.sin(angle))
    cv2.arrowedLine(img, (ix,iy), (ex,ey), (0,0,0), 2, tipLength=0.3)
    cv2.imshow("Demo", img)
    cv2.waitKey(1)
  return act

#----------------------------------------------------------------------------------------

def user_draw_goal():
  cv2.namedWindow('Demo')
  cv2.setMouseCallback('Demo', mouse_callback)
  img = np.zeros((300,300), np.uint8)
  while not down:
    img = copy.deepcopy(img)
    cv2.imshow("Demo", img)
    cv2.waitKey(1)

  last_x = tipx
  last_y = tipy
  while down:
    img = copy.deepcopy(img)

    cv2.line(img, (last_x,last_y), (tipx,tipy), (255,255,255), 10)
    last_x = tipx
    last_y = tipy
    cv2.imshow("Demo", img)
    cv2.waitKey(1)
  img = cv2.resize(img, (64,64))
  return img


#========================================================================================
# Test
#========================================================================================

if __name__ == "__main__":
    env = SimRopeEnv()
    while True:
        obs = env.reset(randomize_start=True)
        done = False
        while not done:
            act = env.random_action()
            obs, rew, done, _ = env.step(act)

            cv2.imshow("img", env.get_image())
            cv2.imshow("obs", obs)
            cv2.waitKey(0)
