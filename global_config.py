from math import pi

terrain_choice = 4

command_with_target_pos = True
fixed_target_pos = True
target_pos = [10, 10]

fixed_target_heading = True
target_heading_deg = 90
target_heading = target_heading_deg * pi / 180
# 右为0度，逆时针旋转

fixed_target_lin_vel = True
target_lin_vel = [1.0, -1.0]
# x=1为上，y=1为左

reward_scale_tracking_target = 1
reward_target_sigma = 2
