import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

mi = 0.3
N = 30000
model = hmm.CategoricalHMM(n_components=3, random_state=99)
model.startprob_ = np.array([1.0, 0, 0])
model.transmat_ = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
model.emissionprob_ = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
feature_matr, gen_states = model.sample(N)

print("Limiting distribution i think: " + str(model.get_stationary_distribution()))
_, ax = plt.subplots()
ax.plot(gen_states[:500])
ax.set_title('States over time')
ax.set_xlabel('Time (# of jumps)')
ax.set_ylabel('State')
plt.xlim([0, 100])
plt.show()

_, ax = plt.subplots()
ax.hist(feature_matr[gen_states == 0], label='State 0 -> state 1', alpha=0.5,
        bins=np.arange(4) - 0.5, density=True)
ax.hist(feature_matr[gen_states == 1], label='State 1 -> state 0\nState 1 -> state 2', alpha=0.5,
        bins=np.arange(4) - 0.5, density=True)
ax.hist(feature_matr[gen_states == 2], label='State 2 -> state 1', alpha=0.5,
        bins=np.arange(4) - 0.5, density=True)
ax.set_title('Jumps probabilities by state')
ax.set_xlabel('Changed state')
ax.set_ylabel('Probability')
ax.legend()
plt.show()
