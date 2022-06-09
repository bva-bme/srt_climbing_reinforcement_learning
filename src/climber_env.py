#!/usr/bin/env python3
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import load_model_from_xml
import os

class Climber(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description
    This environment is derived from the OpenAI gym Humanoid-v3 environment.
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v3.py
    The 3D model of the humanoid is hanged to a vertical finite-element rope on two points: the abdomen (Kroll)
    and its left hand (Poignee). The right hand also holds the Poignee. The Poignee and the left leg of the human
    is connected via a footloop (a rope). The two ascenders (Kroll and Poignee) slide on the main rope upwards
    but self-lock downwards, therefore the agent can move only upwards with a squatting motion.
    The goal of this environment is to fully ascend the rope (8 meters).

    ### Action Space
    The agent take a 17-element vector for actions.
    The action space is a continuous `(action, ...)` all in `[-1, 1]`, where `action`
    represents the numerical torques applied at the hinge joints.
    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|----------------------|---------------|----------------|---------------------------------------|-------|------|
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4 | 0.4 | hip_1 (front_left_leg)      | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4 | 0.4 | angle_1 (front_left_leg)    | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4 | 0.4 | hip_2 (front_right_leg)     | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4 | 0.4 | right_hip_x (right_thigh)   | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4 | 0.4 | right_hip_z (right_thigh)   | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4 | 0.4 | right_hip_y (right_thigh)   | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4 | 0.4 | right_knee                  | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4 | 0.4 | left_hip_x (left_thigh)     | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4 | 0.4 | left_hip_z (left_thigh)     | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4 | 0.4 | left_hip_y (left_thigh)     | hinge | torque (N m) |
    | 10   | Torque applied on the rotor between the left hip/thigh and the left shin          | -0.4 | 0.4 | left_knee                   | hinge | torque (N m) |
    | 11   | Torque applied on the rotor between the torso and right upper arm (coordinate -1) | -0.4 | 0.4 | right_shoulder1             | hinge | torque (N m) |
    | 12   | Torque applied on the rotor between the torso and right upper arm (coordinate -2) | -0.4 | 0.4 | right_shoulder2             | hinge | torque (N m) |
    | 13   | Torque applied on the rotor between the right upper arm and right lower arm       | -0.4 | 0.4 | right_elbow                 | hinge | torque (N m) |
    | 14   | Torque applied on the rotor between the torso and left upper arm (coordinate -1)  | -0.4 | 0.4 | left_shoulder1              | hinge | torque (N m) |
    | 15   | Torque applied on the rotor between the torso and left upper arm (coordinate -2)  | -0.4 | 0.4 | left_shoulder2              | hinge | torque (N m) |
    | 16   | Torque applied on the rotor between the left upper arm and left lower arm         | -0.4 | 0.4 | left_elbow                  | hinge | torque (N m) |

    ### Observation Space
    The state space consists of positional values of different body parts of the Humanoid,
    followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.
    **Note:** The x- and y-coordinates of the torso are being omitted to produce position-agnostic behavior in policies
    The observation is a `ndarray` where the elements correspond to the following:
    | Num | Observation                                                        | Min                | Max                | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------|----------------|-----------------|----------------------------------------|-------|------|
    | 0   | z-coordinate of the torso (centre)                                                                              | -Inf                 | Inf                | root      | free | position (m) |
    | 1   | x-orientation of the torso (centre)                                                                             | -Inf                 | Inf                | root      | free | angle (rad) |
    | 2   | y-orientation of the torso (centre)                                                                             | -Inf                 | Inf                | root      | free | angle (rad) |
    | 3   | z-orientation of the torso (centre)                                                                             | -Inf                 | Inf                | root      | free | angle (rad) |
    | 4   | w-orientation of the torso (centre)                                                                             | -Inf                 | Inf                | root       | free | angle (rad) |
    | 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf                 | Inf               | abdomen_z | hinge | angle (rad) |
    | 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf                 | Inf               | abdomen_y | hinge | angle (rad) |
    | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf                 | Inf               | abdomen_x | hinge | angle (rad) |
    | 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf                 | Inf               | right_hip_x | hinge | angle (rad) |
    | 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf                 | Inf               | right_hip_z | hinge | angle (rad) |
    | 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf                 | Inf               | right_hip_y | hinge | angle (rad) |
    | 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf                 | Inf               | right_knee | hinge | angle (rad) |
    | 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf                 | Inf               | left_hip_x | hinge | angle (rad) |
    | 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf                 | Inf               | left_hip_z | hinge | angle (rad) |
    | 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf                 | Inf               | left_hip_y | hinge | angle (rad) |
    | 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf                 | Inf               | left_knee | hinge | angle (rad) |
    | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf                 | Inf               | right_shoulder1 | hinge | angle (rad) |
    | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf                 | Inf               | right_shoulder2 | hinge | angle (rad) |
    | 18  | angle between right upper arm and right_lower_arm                                                               | -Inf                 | Inf               | right_elbow | hinge | angle (rad) |
    | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf                 | Inf               | left_shoulder1 | hinge | angle (rad) |
    | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf                 | Inf               | left_shoulder2 | hinge | angle (rad) |
    | 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf                 | Inf               | left_elbow | hinge | angle (rad) |
    | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf                 | Inf                | root      | free | velocity (m/s) |
    | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf                 | Inf                | root      | free | velocity (m/s) |
    | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf                 | Inf                | root      | free | velocity (m/s) |
    | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf                 | Inf                | root      | free | anglular velocity (rad/s) |
    | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf                 | Inf                | root      | free | anglular velocity (rad/s) |
    | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf                 | Inf                | root      | free | anglular velocity (rad/s) |
    | 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf                 | Inf               | abdomen_z | hinge | anglular velocity (rad/s) |
    | 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf                 | Inf               | abdomen_y | hinge | anglular velocity (rad/s) |
    | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf                 | Inf               | abdomen_x | hinge | aanglular velocity (rad/s) |
    | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf                 | Inf               | right_hip_x | hinge | anglular velocity (rad/s) |
    | 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf                 | Inf               | right_hip_z | hinge | anglular velocity (rad/s) |
    | 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf                 | Inf               | right_hip_y | hinge | anglular velocity (rad/s) |
    | 35  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf                 | Inf               | right_knee | hinge | anglular velocity (rad/s) |
    | 36  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf                 | Inf               | left_hip_x | hinge | anglular velocity (rad/s) |
    | 37  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf                 | Inf               | left_hip_z | hinge | anglular velocity (rad/s) |
    | 38  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf                 | Inf               | left_hip_y | hinge | anglular velocity (rad/s) |
    | 39  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf                 | Inf               | left_knee | hinge | anglular velocity (rad/s) |
    | 40  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf                 | Inf               | right_shoulder1 | hinge | anglular velocity (rad/s) |
    | 41  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf                 | Inf               | right_shoulder2 | hinge | anglular velocity (rad/s) |
    | 42  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf                 | Inf               | right_elbow | hinge | anglular velocity (rad/s) |
    | 43  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf                 | Inf               | left_shoulder1 | hinge | anglular velocity (rad/s) |
    | 44  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf                 | Inf               | left_shoulder2 | hinge | anglular velocity (rad/s) |
    | 45  | angular velocity of the angle between left upper arm and left_lower_arm                                        | -Inf                 | Inf               | left_elbow | hinge | anglular velocity (rad/s) |


    ### Rewards
    The agent gets a reward each time step based on how much it could ascend.

    ### Episode Termination
    The episode terminates when any of the following happens:
    1. The episode duration reaches a max_step = 5000 timesteps
    2. Any of the state space values is no longer finite
    3. Agent reached the top
    """

    def __init__(self, max_step=5000, dest_height=8):

        EnvName = f"{os.getcwd()}/climber.xml"
        with open(EnvName, 'r') as f:
            model_str = f.read()
        # Geometries
        self.model = load_model_from_xml(model_str)
        self._rope_low_geom_id = self.model.geom_names.index('G47')
        self._rope_high_geom_id = self.model.geom_names.index('G1')
        self._kroll_geom_id = self.model.geom_names.index('kroll')
        self._poignee_geom_id = self.model.geom_names.index('poignee')
        self._rope_segment_length = self.model.geom_size[self._rope_low_geom_id][1] * 2

        self.reward = 0
        self.curr_step = 0
        self.prev_height = 1.3
        self.max_step = max_step
        self.dest_height = dest_height
        self.human_joints = ['root',
                             'abdomen_z',
                             'abdomen_y',
                             'abdomen_x',
                             'right_hip_x',
                             'right_hip_z',
                             'right_hip_y',
                             'right_knee',
                             'left_hip_x',
                             'left_hip_z',
                             'left_hip_y',
                             'left_knee',
                             'right_shoulder1',
                             'right_shoulder2',
                             'right_elbow',
                             'left_shoulder1',
                             'left_shoulder2',
                             'left_elbow']

        mujoco_env.MujocoEnv.__init__(self, EnvName, 5)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        data = self.sim.data
        observed_states = [data.get_joint_qpos('root').flat[2:]]

        # Positions
        for joint in self.human_joints:
            if joint == 'root':
                pass
            else:
                observed_states.append(data.get_joint_qpos(joint).flat)

        # Velocities
        for joint in self.human_joints:
            observed_states.append(data.get_joint_qvel(joint).flat)

        return np.concatenate(observed_states)

    # TODO
    def _calculate_reward(self, data):
            self.reward = (data.get_joint_qpos('root').flat[2] - self.prev_height) * 1000  # upwards (ascend per step)


    def _get_termination(self, data):
        # Ran for too many steps
        height = data.get_joint_qpos('root')[2]
        if self.curr_step >= self.max_step:
            print(f'Episode terminated after {self.max_step} steps. Ascended height: {height} m.')
            return bool(True)
        # States inf or nan
        if not np.isfinite(data.qpos).any() or not np.isfinite(data.qvel).any() or not np.isfinite(data.qacc).any():
            print('Episode terminated: environment state Inf or NaN!')
            return bool(True)
        # Success
        if height >= self.dest_height:
            print(f'Episode terminated after the agent successfully ascended {height} m.')
            return bool(True)
        return bool(False)

    def step(self, a):
        self.prev_height = self.sim.data.get_joint_qpos('root').flat[2]
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        self._update_constraints(data)
        self._calculate_reward(data)
        done = self._get_termination(data)

        self.curr_step += 1
        return (
            self._get_obs(),
            self.reward,
            done,
            '',
        )

    def reset_model(self):
        c = 0.0
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        self._init_constraints()
        self.reward = 0
        self.curr_step = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20

    # Euclidean distance of two points
    def _dist(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2 + (a[2] - b[2]) ** 2)

    # Check if the ascender (kroll or poignee) can be constrained the next rope segment
    def _ascend_segment(self, ascender, rope_segment_next, rope_segment_length):
        asc_thd = 2.5
        if self._dist(ascender, rope_segment_next) < rope_segment_length * asc_thd:
            return True
        else:
            return False

    def _update_constraints(self, data):
        # if not yet reached 8 meters
        if self.model.eq_obj2id[1] > self._rope_high_geom_id + 8:
            # kroll
            if self._ascend_segment(data.geom_xpos[self._kroll_geom_id],
                              data.geom_xpos[self.model.eq_obj2id[1] - 1],
                              self._rope_segment_length):
                self.model.eq_obj2id[1] -= 1
                # print("Kroll goes UP!")
            # poignee
            if self._ascend_segment(data.geom_xpos[self._poignee_geom_id],
                              data.geom_xpos[self.model.eq_obj2id[3] - 1],
                              self._rope_segment_length):
                self.model.eq_obj2id[3] -= 1

    def _init_constraints(self):
        # Bottom of the rope
        self.model.eq_obj2id[1] = 124
        self.model.eq_obj2id[3] = 114

