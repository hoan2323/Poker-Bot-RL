# Thêm vào cuối train.py, sau run_training()
# Hoặc chạy riêng trong file check_log.py

import numpy as np
import matplotlib.pyplot as plt

rewards   = np.load('rewards.npy')
win_rates = np.load('win_rates.npy')

window = 10_000
smoothed_wr  = np.convolve(win_rates,  np.ones(window)/window, mode='valid')
smoothed_rew = np.convolve(rewards,    np.ones(window)/window, mode='valid')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(smoothed_wr);  ax1.set_title('WinRate (smoothed 10k)'); ax1.axvline(200_000, color='r', linestyle='--', label='Phase 2'); ax1.axvline(400_000, color='g', linestyle='--', label='Phase 3'); ax1.legend()
ax2.plot(smoothed_rew); ax2.set_title('AvgReward (smoothed 10k)')
plt.tight_layout()
plt.savefig('training_curve.png')
print(f"Total episodes trained: {len(rewards)}")
print(f"Final epsilon estimate: {0.35 * (0.999990 ** len(rewards)):.4f}")
print(f"WinRate last 10k: {win_rates[-1]:.4f}")