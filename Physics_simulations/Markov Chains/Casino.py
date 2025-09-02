import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

np.random.seed(42)

model = hmm.CategoricalHMM(n_components=2, random_state=99)
model.startprob_ = np.array([1.0, 0.0])
model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
model.emissionprob_ = np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6], [1 / 6, 1 / 6, 1 / 6, 0, 0, 0.5]])
rolls, gen_states = model.sample(30000)
print("Limiting distribution i think: " + str(model.get_stationary_distribution()))

fig, ax = plt.subplots()
ax.plot(gen_states[:500])
ax.set_title('States over time')
ax.set_xlabel('Time (# of rolls)')
ax.set_ylabel('State')
plt.xlim([0, 100])
plt.show()

# plot rolls for the fair and loaded states
fig, ax = plt.subplots()
ax.hist(rolls[gen_states == 0], label='fair', alpha=0.5,
        bins=np.arange(7) - 0.5, density=True)
ax.hist(rolls[gen_states == 1], label='loaded', alpha=0.5,
        bins=np.arange(7) - 0.5, density=True)
ax.set_title('Roll probabilities by state')
ax.set_xlabel('Count')
ax.set_ylabel('Roll')
ax.legend()
plt.show()
