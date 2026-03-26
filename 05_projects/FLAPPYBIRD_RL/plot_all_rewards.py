import matplotlib.pyplot as plt

episodes = [1, 3, 11, 216, 505, 1005, 1402, 1991, 2184, 2683, 2694, 3405, 4417, 4788, 5287, 5288, 5404, 5538, 5642, 8809]
rewards = [-8.099999999999998, -7.499999999999998, -3.899999999999998, -1.4999999999999982, 
           -0.8999999999999986, -0.29999999999999893, 2.599999999999998, 2.799999999999999,
           3.1999999999999975, 3.500000000000001, 5.199999999999994, 5.499999999999988,
           5.999999999999988, 6.7999999999999865, 7.599999999999985, 8.099999999999977, 
           8.099999999999984, 9.299999999999974, 11.99999999999996, 18.50000000000001]

# with open("runs/flappybirdv0_all_rewards.log") as f:
#     for line in f:
#         ep, r = line.strip().split(",")
#         episodes.append(int(ep))
#         rewards.append(float(r))

plt.plot(episodes, rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Training Curve")
plt.show()