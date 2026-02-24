import os

filepath = "./logs/go2_stair_exp/Feb06_21-32-35_go2_stair_exp_2/"

for f in os.listdir(filepath):
    if f.startswith("events"):
        filepath = os.path.join(filepath, f)
        break

with open(filepath, "r") as f:
    for line in f:
        print(line)
