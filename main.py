"""
Three-Link Manipulator Environment with Reinforcement Learning (PPO)

Author: Hamid Reza Heidari
HW2 - AI in Robotics

Description:
This code implements a custom OpenAI Gym environment for a 3-link planar manipulator.
The manipulator is controlled to reach a target point using Proximal Policy Optimization (PPO).

Assumptions:
1. Theta1 is the angle between the first arm and the horizontal.
2. Theta2 is the angle between the first and second arm.
3. Theta3 is the angle between the second and third arm.
4. The manipulator is situated at the origin (0,0).
"""

# Install required packages (uncomment if needed)
# !pip install shimmy
# !pip install stable-baselines3[extra]

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
import gym
from gym import spaces
from stable_baselines3 import PPO

# --------------------------
# Forward Kinematics Function
# --------------------------
def forward_kinematic(thetas, link_lengths):
    """
    Calculate the (x, y) coordinates of each joint and the end effector.

    Parameters:
        thetas (list/tuple): [theta1, theta2, theta3] angles in radians
        link_lengths (list/tuple): [L1, L2, L3] lengths of the links

    Returns:
        tuple: Coordinates of joints and end-effector: ((x0,y0),(x1,y1),(x2,y2),(x3,y3))
    """
    L1, L2, L3 = link_lengths
    T1, T2, T3 = thetas

    # Joint positions
    x1, y1 = L1*cos(T1), L1*sin(T1)
    x2, y2 = x1 + L2*cos(T1 + T2), y1 + L2*sin(T1 + T2)
    x3, y3 = x2 + L3*cos(T1 + T2 + T3), y2 + L3*sin(T1 + T2 + T3)

    return (0, 0), (x1, y1), (x2, y2), (x3, y3)

# --------------------------
# Custom Gym Environment
# --------------------------
class ThreeLinkManipulatorEnv(gym.Env):
    """
    Custom OpenAI Gym environment for a 3-link planar manipulator.
    The goal is to move the end-effector to a target point.
    """

    def __init__(self):
        super().__init__()

        # Link lengths
        self.link_lengths = [1.0, 1.0, 1.0]

        # Initial joint angles and target position
        self.initial_state = [0.0, 0.0, 0.0]
        self.target_point = [1.6, 2.0]

        # Action: small changes in joint angles
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32)

        # Observation: joint angles
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)

        # Hyperparameters for reward
        self.alpha = 0.4  # weight for joint angles penalty
        self.beta = 0.6   # weight for action magnitude penalty

        # Current joint angles
        self.state = self.initial_state.copy()

    def reset(self):
        """Reset the environment to the initial state."""
        self.state = self.initial_state.copy()
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """
        Apply action to the environment and return the next state, reward, done, info.

        Parameters:
            action (ndarray): Changes in joint angles

        Returns:
            state (ndarray): New joint angles
            reward (float): Reward for the action
            done (bool): True if target is reached
            info (dict): Additional info (empty)
        """
        # Update joint angles and clip to [-pi, pi]
        self.state = np.clip(np.array(self.state) + action, -np.pi, np.pi)

        # Forward kinematics to find end-effector position
        fk = forward_kinematic(self.state, self.link_lengths)
        end_effector = fk[3]

        # Distance to target
        distance_to_target = np.linalg.norm(np.array(end_effector) - np.array(self.target_point))

        # Reward function
        reward = 1 / (distance_to_target + self.alpha*np.sum(np.abs(self.state)) + self.beta*np.sum(np.square(action)))

        # Check if task is done
        done = distance_to_target < 0.0005

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        """Visualize the manipulator and target point."""
        fk = forward_kinematic(self.state, self.link_lengths)
        x_coords, y_coords = zip(*fk)

        plt.figure()
        plt.plot([x_coords[0], x_coords[1]], [y_coords[0], y_coords[1]], 'bo-', lw=3)
        plt.plot([x_coords[1], x_coords[2]], [y_coords[1], y_coords[2]], 'go-', lw=3)
        plt.plot([x_coords[2], x_coords[3]], [y_coords[2], y_coords[3]], 'mo-', lw=3)
        plt.plot(self.target_point[0], self.target_point[1], 'rx', markersize=8)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid(True)
        plt.legend(["Link1", "Link2", "Link3", "Target", f"alpha={self.alpha}", f"beta={self.beta}"], loc='lower left')
        plt.show()

# --------------------------
# Training PPO Model
# --------------------------
env = ThreeLinkManipulatorEnv()
model = PPO("MlpPolicy", env, verbose=1)

# Train for multiple iterations
for _ in range(15):
    model.learn(total_timesteps=10000, progress_bar=True)

# --------------------------
# Test the Trained Model
# --------------------------
obs = env.reset()
for _ in range(250):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()
