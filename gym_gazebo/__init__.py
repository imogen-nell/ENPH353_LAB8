import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# cart pole
register(
    id='GazeboCartPole-v0',
    entry_point='gym_gazebo.envs.gazebo_cartpole:GazeboCartPolev0Env',
)

register(
	id='Gazebo_linefollow-v0',
	entry_point='gym_gazebo.envs.gazebo_linefollow:Gazebo_Linefollow_Env',
	max_episode_steps=3000,
)