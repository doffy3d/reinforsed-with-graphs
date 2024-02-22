from environment import Env
import numpy as np
from matplotlib import pyplot as plt

# %%

class Actor:
    def __init__(self, env, gamma=0.9, eps=0.5):
        self.state = 8
        self.Q = np.zeros((4, 12))
        self.gamma = gamma
        self.eps = eps
        self.env = env

    def setConstAlpha(self, alpha):
        self.alpha = lambda t: alpha
        self.type = 'constant alpha = ' + str(alpha)

    def setAdaptAlpha(self):
        self.alpha = lambda t: np.log(t + 1) / (t + 1)
        self.type = 'adaptive alpha = log(t+1)/(t+1)'

    def go(self, tol=40):
        self.t = 1
        opt_pol_ct = 0
        v_plt = []

        while self.t < 5000:
            finished = self.move()
            if finished:
                self.t += 1
                v_plt.append(np.max(self.Q, axis=0))
                if (self.env.optimalPolicy() == self.optimalPolicy()).all():
                    opt_pol_ct += 1
                else:
                    opt_pol_ct = 0

                if opt_pol_ct > tol:
                    break

        v_plt = np.array(v_plt)
        env_V = np.max(self.env.Q, axis=0)
        plt.figure()
        ax = plt.gca()
        for i in range(v_plt.shape[1]):
            if i in [3, 5, 7]:
                continue
            """"
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(v_plt[:, i], label=str(self.env.toMatrix(i)), color=color)
            plt.scatter(self.t, env_V[i], color=color, label=None)
            """""
            # Choose a color for plotting
            color = plt.cm.viridis(i / v_plt.shape[1])  # Example using the Viridis colormap
            plt.plot(v_plt[:, i], label=str(self.env.toMatrix(i)), color=color)
            plt.scatter(self.t, env_V[i], color=color, label=None)

        plt.legend(loc='upper left')
        plt.title('V vrednosti po epizodama za eps = ' + str(self.eps) + ', ' + self.type)
        plt.xlabel('Broj epizode')
        plt.close()
        return self.t

    def move(self):
        pickbest = np.random.random() > self.eps
        action = np.argmax(self.Q[:, self.state]) if pickbest else np.random.choice(4)
        tmp = self.env.sim(self.state, action)
        ns = tmp[0]
        r = tmp[1]

        if ns == -1:
            self.Q[action, self.state] = r
            self.state = 8
            return True
        qval = r + self.gamma * np.max(self.Q[:, ns])
        self.Q[action, self.state] += self.alpha(self.t) * (qval - self.Q[action, self.state])
        self.state = ns
        return False

    def optimalPolicy(self):
        return np.argmax(self.Q, axis=0)


p = 0.8
env = Env(p)
env.initQ()
env.iterQ(0.9)

# %%
actor = Actor(env, gamma=0.9, eps=0.5)
np.set_printoptions(precision=2)
actor.setConstAlpha(0.1)
actor.setAdaptAlpha()
print(actor.go())

# print(actor.Q)
# print(env.Q)

print(actor.optimalPolicy())
print(env.optimalPolicy())

# %% Gamma convergence experiment
for gamma in [1, 0.9]:
    env.initQ()
    print(env.iterQ(gamma))

# %% Success rate experiment
print('Originalna politika')
print(env.optimalPolicy())

env2 = Env(0.4)
env2.initQ()
env2.iterQ(0.9)
print('Politika za p = 0.4')
print(env2.optimalPolicy())

# %% Epsilon and Alpha
for epsilon in [0.4, 0.7, 0.9]:
    for alpha in [0.01, 0.05, 0.1, 0.5]:
        actor = Actor(env, gamma=0.9, eps=epsilon)
        actor.setConstAlpha(alpha)
        actor.go()

        if alpha == 0.1:
            actor.setAdaptAlpha()
            actor.go()



