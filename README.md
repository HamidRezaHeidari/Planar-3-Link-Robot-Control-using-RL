# 📖Robotic Arm Simulation & Reinforcement Learning

In this project, we are going to control it by simulating a 3-degree-of-freedom robot with deep reinforcement learning methods. let's pay First, it is assumed that we have a screen robot and we want the final execution of the robot arm from the starting point to the target point. The robot is trained to move its end-effector from a fixed start position to a fixed target position with high precision.The environment combines forward kinematics, custom reward shaping, and PPO (Proximal Policy Optimization) to allow the agent to learn efficient, smooth, and accurate joint 

---

## 💡Features

-Fully simulated 3-link planar manipulator
-Reinforcement learning-based control using Stable Baselines3 PPO
-Custom reward function balancing distance to target, joint movement penalties, and action magnitude
-Visual rendering of robot motion using Matplotlib
-Easy-to-extend environment for other RL experiments

## 🧭About Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards for good actions and penalties for bad ones, gradually learning an optimal policy to achieve a goal. In this project, RL trains the robot arm to reach the target efficiently and accurately.
<br/>
✨Star if you find it useful
  <br />
<p align="center">
  <img title="Fig1" height="400" src="images/1.gif">
  <br />
</p>



