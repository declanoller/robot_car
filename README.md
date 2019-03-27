
Robot Reinforcement Learning Project
=========================================

<p align="center">
  <img width="400" height="400" src="https://github.com/declanoller/robot_car/blob/master/other/robo_blog_cover.png">
</p>

Overview
-----------------------

This is the set of programs and classes I used to have a robot learn the game usually called "puckworld", using Q learning with experience replay. It probably won't be directly useful to anyone else (it has a lot of things specific to my setup, like sensors and geometry), but it could probably be easily modified.



Main run scripts
--------------------

These are the main `.py` scripts that I use to run the robot in various ways, in the main directory for simplicity. They are:

* `learn.py` - For learning over the course of many episodes that can be interrupted and resumed. Uses `RobotTools.py`.
* `drive.py` - For simply driving the robot around in real time.
* `robot_arena_control.py` - For controlling the robot within the arena, with a CLI GUI. It uses all sensor information as the learning will do, but you control it directly.
* `model_run.py` - For loading a pre-trained model, usually used for loading a computer trained model to just see the robot working successfully.
* `plot_progress.py` - For inspecting the progress of the robot remotely as it runs. It uses `scp` to fetch the `resume.json` file from the robot for the current run. Then, if it's missing any of the most recent data, it fetches that as well, and plots the movement history, average reward history, and representative plots of the current Q function.



Classes
-----------------------

These are the classes that the run scripts use, which live in the classes directory. I'll list the more "software" ones and then the "hardware" ones that interface with sensors, etc.

Software:

* `Agent.py` - Responsible for the actual RL. Takes an agent class, in this case `Robot.py`.
* `Robot.py` - Controls the robot, by interfacing with all the sensor classes.
* `RobotTools.py` - This is for setting up a learning run that will periodically save all the progress, so you can pick back up again if the robot randomly stops working, you can stop it to change something and resume, or to inspect the evolution of the learning process.
* `DebugFile.py` - This is for logging everything that the robot does, in a format I like. However, since it logs so much, it will produce log files that are 10s of MB large over a full training run, and therefore I only enable it for short runs of debugging some problem.
* `FileSystemTools.py` - Just a little list of custom functions that I found myself using often, mostly string formatting type stuff.

Hardware:

* `Compass.py` - Interfaces with the popular MPU9250/6500 gyrometer/accelerometer/compass chip, which I use for only the compass. It basically just leverages richardstechnotes' amazing RTIMULib2 code (https://github.com/richardstechnotes/RTIMULib2) and tweaks the output a little.
* `Motor.py` - For giving signals to the L298N motor driver I'm using. There are some tweaks to account for moving on different surfaces and such.
* `MQTTComm.py` - For setting up a MQTT server that the ESP-12 that interfaces with the IR sensors (for the targets) will communicate with.
* `TOF.py` - For interfacing with the VL53LOX TOF (time-of-flight) sensors the robot uses to measure distances. Uses a VL53LOX library such as (https://github.com/johnbryanmoore/VL53L0X_rasp_python), though there appear to be many copies at this point and I'm not sure who did the original.



















#
