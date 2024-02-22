from matplotlib import pyplot as plt
import numpy as np

FIELD_WIDTH = 4
FIELD_HEIGHT = 3
FIELD_SIZE = 12
UNAVAILABLE = 5


class Env:

    def __init__(self, p):
        '''
        -------------------
        | 0 | 1 | 2  | 3  |
        -------------------
        | 4 | 5 | 6  | 7  |
        -------------------
        | 8 | 9 | 10 | 11 |
        -------------------

        5 - Unavailable
        8 - Start

        3 - Terminal, +1
        7 - Terminal, -1

        Parameters
        ----------
        p : Success rate [0,1]

        '''
        # self.table = np.zeros((FIELD_HEIGHT, FIELD_WIDTH))
        # initialize rewards
        self.R = np.ones((FIELD_HEIGHT, FIELD_WIDTH)) * (-0.04)
        self.R[0][3] = 1
        self.R[1][3] = -1

        # helper functions
        self.toArray = lambda i, j: i * FIELD_WIDTH + j
        self.toMatrix = lambda k: (k // FIELD_WIDTH, k % FIELD_WIDTH)

        # movement probabilities
        self.success = p
        self.slip = (1 - p) / 2

    def initQ(self):
        '''

        Actions:
        0 - up
        1 - right
        2 - down
        3 - left
        If on terminal square, no actions allowed

        '''
        self.Q = np.zeros((4, FIELD_SIZE))
        self.neighbours = np.zeros((4, FIELD_SIZE), dtype='i')
        for k in range(FIELD_SIZE):
            if k in [UNAVAILABLE, 3, 7]:
                self.neighbours[:, k] = -1
                continue
            i, j = self.toMatrix(k)
            self.neighbours[:, k] = [self.toArray(i - 1, j), self.toArray(i, j + 1),
                                     self.toArray(i + 1, j), self.toArray(i, j - 1)]
            if j == 0:
                self.neighbours[3, k] = k
            if j == FIELD_WIDTH - 1:
                self.neighbours[1, k] = k
            if i == 0:
                self.neighbours[0, k] = k
            if i == FIELD_HEIGHT - 1:
                self.neighbours[2, k] = k
            self.neighbours[:, k][self.neighbours[:, k] == UNAVAILABLE] = k

    def iterQ(self, gamma, tol=1E-5):
        op_plt = []
        q_plt = []
        np.set_printoptions(precision=2)
        for ct in range(1000):
            newQ = np.copy(self.Q)
            state_max = np.max(self.Q, axis=0)
            for i in range(4):
                Psa = self.getP(i)
                for k in range(FIELD_SIZE):
                    newQ[i, k] = self.R[self.toMatrix(k)]
                    if sum(self.neighbours[:, k]) == -4:
                        continue
                    newQ[i, k] += gamma * (
                        sum(np.multiply(Psa, state_max[self.neighbours[:, k]])))
            op_plt.append(self.optimalPolicy())
            q_plt.append(np.max(self.Q, axis=0))

            if max(abs(newQ - self.Q).reshape(-1)) < tol:
                break

            self.Q = newQ
            # print(self.Q)
        op_plt = np.array(op_plt)
        plt.figure()
        for i in range(op_plt.shape[1]):
            if i in [3, 5, 7]:
                continue
            plt.plot(op_plt[:, i], label=str(self.toMatrix(i)))
        plt.title('Upravljanje po iteracijama za gamma = ' + str(gamma))
        plt.xlabel('Broj iteracije')
        plt.legend(loc='right')

        plt.figure()
        q_plt = np.array(q_plt)
        for i in range(q_plt.shape[1]):
            if i in [3, 5, 7]:
                continue
            plt.plot(q_plt[:, i], label=str(self.toMatrix(i)))
        plt.title('V vrednosti po iteracijama za gamma = ' + str(gamma))
        plt.xlabel('Broj iteracije')
        plt.legend(loc='right')
        return ct

    def getP(self, i):
        Psa = np.zeros(4)
        Psa[i] = self.success
        Psa[(i + 1) % 4] = self.slip
        Psa[(i - 1) % 4] = self.slip
        return Psa

    def sim(self, state, action):
        new_state = int(np.random.choice(self.neighbours[:, state],
                                         p=self.getP(action)))
        reward = self.R[self.toMatrix(state)]
        return (new_state, reward)

    def optimalPolicy(self):
        return np.argmax(self.Q, axis=0)

