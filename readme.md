# Bipedal Robot 
A bipedal robot from scratch except actuators


### CAD
![CAD model](docs/cad_model.png)

### Simulation
![MuJoCo simulation](docs/mujoco_sim.gif)



## Results

### Walking (ZMP) 
![Sim2Real walking](docs/sim2real_walk_line.gif)

### Jump - the Whole-body Control with full kinematics model

![Jump](docs/wbmp_jump.gif)

It acheives 10 mm height jump.


To get higher jump, it needs actuators that have faster a angular velocity.



# Until Now

- Made a simple bipedal robot using Fusion 360.
- Spawned the robot in MuJoCo.
- Walking using ZMP and Jump using WBC-full kinematics.
- Assembled the robot in real.
- Implemented ZMP-based walking.


# Next Steps
- Implement a walking algorithm based on Reinforcement Learning
- Implement a walking algorithm based on Model-based control
- How about hybrid system with RL and Model-based?

# Challenges

- Jump and land <- This will be a main challenge, but the angular velocity of actuators are not enough. 
- Consecutive jumps <- Also a challenge
- Maybe Backflip..? <- I am not sure if the robot has enough angular velocity to do this...


# Reference
- Robotis OP3: https://github.com/robotis-git/robotis-op3
- Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics, 2014, doi: [10.1109/HUMANOIDS.2014.7041375](https://doi.org/10.1109/HUMANOIDS.2014.7041375).



