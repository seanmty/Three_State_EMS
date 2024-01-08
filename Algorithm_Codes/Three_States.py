import random
import time
import math
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import comb
from block_prob import *

def Get_Effective_Lambda(L, Mu, N):
    """
    Get the effective lambda that gives the desired offered load
    L is calculated by rho_total, which is carried load here
    Using Newton's Method
    :param L: np.sum(rho_i + (1-rho_i)*alpha)
    :param Mu:
    :param N:
    :return: Lambda_eff
    """
    rho_ = 0
    rho = 1  # this is not use Just an initialization that will be replaced by rho_ in the iteration
    step = 0
    while np.abs(rho - rho_) > 0.001:
        rho = rho_
        l = rho * Mu
        B = ErlangLoss(l, Mu, N)[-1]  # the loss probability for the loss system
        rho_ = rho - (rho * (1 - B) - L) / (1 - B - (N - rho + rho * B) * B)
        step += 1
        if step > 100:
            break
    Lambda_eff = rho * Mu
    return Lambda_eff

class Two_State_Hypercube():
    def __init__(self, data_dict=None):
        # initilize data stored in self.data_dict
        self.keys = ['N', 'K', 'Lambda', 'Mu', 'Mu_vec', 'Mu_mat', 'frac_j', 't_mat', 'pre_list', 'pol']
        self.rho_hyper, self.rho_approx, self.rho_simu = None, None, None  # initialize the utilizations to be None
        self.prob_dist = None
        self.data_dict = dict.fromkeys(self.keys, None)
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys:
                    self.data_dict[k] = v
        self.G = None  # G for approximation
        self.Q = None  # Q for approximation
        self.r = None  # r for approximation
        self.P_b = None  # This is to differentiate from 3-state case
        self.q_nj = None  # This is the dispatch probability using Linear-alph algorithm. This is shared by both 2-state and 3-state algorithms.
        self.time_approx = None
        self.P_n = None

    def Update_Parameters(self, **kwargs):
        # update any parameters passed through kwargs
        for k, v in kwargs.items():
            if k in self.keys:
                self.data_dict[k] = v
                if k == 'pre_list':  # reset G if pre_list changes
                    self.G = None

    def Random_Pref(self, seed=9001):
        # random preference list
        random.seed(seed)  # Set Random Seed
        N, K = self.data_dict['N'], self.data_dict['K']
        # Shuffle the IDs as each preference list
        pre_list = np.array([random.sample(list(range(N)), N) for _ in range(K)])
        self.Update_Parameters(pre_list=pre_list)

    def Random_Fraction(self, seed=9001):
        """
        Obtain random frac.
        """
        np.random.seed(seed)
        K = self.data_dict['K']
        frac_j = np.random.random(size=K)
        frac_j /= sum(frac_j)
        self.data_dict['frac_j'] = frac_j

    def Random_Mu(self, average_mu, radius=0.2, seed=9001):
        """
        This is a function to generate a vector of Mu,
        Mu[i] is the expected service time of unit i.
        The Mu vector is stored in Mu_vec
        :param average_mu: given average service time among units
        :param radius: in (0,1), to control random mu within average_mu*(1 +- radius)
        """
        np.random.seed(seed)
        N = self.data_dict['N']
        Mu_vec = (np.random.rand(N) - 0.5) * radius + 1
        Mu_vec = Mu_vec / np.sum(Mu_vec) * (average_mu * N)
        self.Update_Parameters(Mu_vec=Mu_vec, Mu=average_mu)

    def Random_Mu_nj(self, average_mu, radius=0.2, seed=9001):
        """
            This is a function to generate a vector of Mu,
            Mu[j][i] is the expected service time of unit i at atom j.
            The Mu vector is stored in Mu_vec
            :param average_mu: given average service time among units
        """
        np.random.seed(seed)
        N = self.data_dict['N']
        K = self.data_dict['K']
        Mu_mat = (np.random.rand(K, N) - 0.5) * radius + 1
        Mu_mat = Mu_mat / np.sum(Mu_mat) * (average_mu * N * K)
        self.Update_Parameters(Mu_mat=Mu_mat, Mu=average_mu)

    def Random_Time_Mat(self, t_min=1, t_max=10, seed=9001):
        np.random.seed(seed)
        N, K = self.data_dict['N'], self.data_dict['K']
        t_mat = np.random.uniform(low=t_min, high=t_max, size=(K, N))
        self.Update_Parameters(t_mat=t_mat)

    def Myopic_Policy(self, source='t_mat'):
        # Obtain the policy to dispatch the cloest available unit given the time matrix
        N, K = self.data_dict['N'], self.data_dict['K']
        if source == 't_mat':
            t_mat = self.data_dict['t_mat']
            pre_list = t_mat.argsort(axis=1)
            self.Update_Parameters(pre_list=pre_list)
        elif source == 'pre':
            pre_list = self.data_dict['pre_list']
        else:
            print('Wrong source!')
        policy = np.zeros([2 ** N, K], dtype=int)
        for s in range(2 ** N):
            for j in range(K):
                pre = pre_list[j]
                for n in range(N):
                    if not s >> pre[n] & 1:  # n th choice is free
                        policy[s, j] = pre[n]
                        break
        self.data_dict['pol'] = policy

    def Cal_Trans(self, vec_flag):
        """
        Calculate transition matrix for two state system
        :param vec_flag: True if Mu is heterogeneous among units; False if homogeneous
        :return: transition matrix
        """
        if vec_flag:
            keys = ['N', 'K', 'Lambda', 'Mu_vec', 'pol', 'frac_j']
        else:
            keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Calculate the
        N_state = 2 ** N
        A = np.zeros([N_state, N_state])
        # Calculate upward transtition
        for s in range(N_state - 1):  # The last state will not transition to other states by a arrival
            pol_s = pol[s]
            for j in range(K):
                dis = pol_s[j]
                A[s, s + 2 ** dis] += Lambda * frac_j[j]
        # Calculate downward transtition
        for s in range(1, N_state):  # The first state will not transition
            bin_s = bin(s)
            len_bin = len(bin_s)
            i = 0
            while bin_s[len_bin - 1 - i] != 'b':
                if bin_s[len_bin - 1 - i] == '1':
                    if vec_flag:
                        A[s, s - 2 ** i] = Mu[i]
                    else:
                        A[s, s - 2 ** i] = Mu
                i += 1
        return A

    def Solve_Hypercube(self, update_rho=True):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]

        # Get the transition Matrix
        A = self.Cal_Trans(self.data_dict['Mu_vec'] is not None)
        # Solve for the linear systems
        transition = A.T - np.diag(A.T.sum(axis=0))
        transition[-1] = np.ones(2 ** N)
        b = np.zeros(2 ** N)
        b[-1] = 1
        start_time = time.time()  # staring time
        prob_dist = np.linalg.solve(transition, b)
        print("------ Hypercube run %s seconds ------" % (time.time() - start_time))
        if update_rho:  # store the utilizations
            statusmat = [("{0:0" + str(N) + "b}").format(i) for i in range(2 ** N)]
            busy = [[N - 1 - j for j in range(N) if i[j] == '1'] for i in statusmat]
            rho = [sum([prob_dist[j] for j in range(2 ** N) if i in busy[j]]) for i in range(N)]
            self.rho_hyper = rho
        self.prob_dist = prob_dist
        return prob_dist

    def Get_MRT_Hypercube(self):  # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pol', 'frac_j', 't_mat']
        N, K, pol, frac_j, t_mat = [self.data_dict.get(key) for key in keys]
        prob_dist = self.prob_dist
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for n in range(N):  # The last state has value 0
            q_nj[:, n] = frac_j * np.dot(prob_dist[:-1], pol[:-1, :] == n)  # here we don't need last state so take :-1
        q_nj /= (1 - prob_dist[-1])
        self.q_nj = q_nj  # store these values in the class
        MRT = np.sum(q_nj * t_mat)
        MRT_j = np.sum(q_nj * t_mat, axis=1) / np.sum(q_nj, axis=1)
        return MRT, MRT_j

    # For Approximation
    def Cal_P_n(self):
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        P_n = ErlangLoss(Lambda, Mu, N)
        return P_n

    def Cal_Q(self, P_n=None, r=None):
        """
        Calculate modification factor Q[1,...,k,...]
        :param P_n: Probability distribution
        :param r: average utilization
        :return: Q
        """
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        if self.G is None:
            self.G = [[np.where(self.data_dict['pre_list'][:, i] == j)[0] for i in range(N)] for j in range(N)]
        Q = np.zeros(N)
        if P_n is None:
            P_n = self.Cal_P_n()
        N = len(P_n) - 1
        if r is None:
            r = np.dot(P_n, np.arange(N + 1)) / N
        for j in range(N):
            Q[j] = sum([math.factorial(k) / math.factorial(k - j) * math.factorial(N - j) / math.factorial(N) * (
                    N - k) / (N - j) * P_n[k] for k in range(j, N)]) / (r ** (j) * (1 - r))
        self.Q = Q
        self.r = r
        return Q

    def Two_State_Approx(self, normalize=False, use_effective_lambda=True, epsilon=0.0001, flag_diff_mu=0):
        """
        Approximate utilization for each subsystem,
        alpha is considered when the other subsystem exists.
        :param normalize: In our method, normalization is relaxed
        :param use_effective_lambda: The method we propose
        :param epsilon:
        :param flag_diff_mu: 1 for heterogeneous mu among units, 0 for homogeneous
        :return: approximated utilization
        """
        keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 'pre_list']
        N, K, Lambda, Mu, frac_j, pre_list = [self.data_dict.get(key) for key in keys]
        if flag_diff_mu == 1:
            Mu_vec = self.data_dict.get('Mu_vec')
        try:
            alpha = self.alpha
        except:
            #print('Two state!')
            alpha = 0

        # Step 0: Initialization
        self.Cal_Q()  # This calculates Q, r, and G
        r = self.r  # average fraction of busy time for each unit in the system. Calculated in Cal_Q
        if normalize:
            # P_b is the block probability. The probability that all units are busy
            # This is only calculated then normalize=True (Hua's method 2022)
            r = Lambda / (N * Mu) * (1 - self.P_b)
        rho_i = np.full(N, r)  # utilization of each unit i. Initialization
        rho_i_ = np.zeros(N)  # temporary utilizations to store new value at each step
        n = 0
        # Step 1: Iteration
        start_time = time.time()
        while True:
            n += 1  # increase step by 1
            rho_total = (rho_i + (1 - rho_i) * alpha)  # rho_i denotes rho^{y}, rho_total denotes the total rho
            ######################
            # Use the effective lambda to get the most accurate P_n and Q for each iteration (This helps a lot)
            if use_effective_lambda:
                L = rho_total.sum()
                Lambda_eff = Get_Effective_Lambda(L, Mu, N)
                P_n = ErlangLoss(Lambda_eff, Mu, N)
                self.P_n = P_n
                self.Cal_Q(P_n=P_n, r=L/N)  # when use this Q, the old Q is overwritten and does not work
            ######################
            for i in range(N):  # for each unit
                value = 1  #
                for k in range(N):  # for each order
                    prod_g_j = 0  # Product term for each sum term
                    for j in self.G[i][k]:
                        prod_g_j += Lambda * frac_j[j] * self.Q[k] * np.prod(
                            rho_total[pre_list[j, :k]])  # if alpha = 0, this is just rho in paranthasis
                    if flag_diff_mu == 1:
                        value += (1 / Mu_vec[i]) * prod_g_j  # There should be a 1/mu here. because we don't assume it to be 1.
                    else:
                        value += (1 / Mu) * prod_g_j
                rho_i_[i] = (1 - ((1 - rho_i) * alpha)[i]) * (
                        1 - 1 / value)  # again when alpha = 0, this is 1 in the paranthesis
            # If in PIA condition, use following codes
            # for i in range(N):  # for each unit i
            #     value = 1
            #     for j in range(K):  # for each atom j
            #         if i in pre_list[j]:
            #             k = np.where(pre_list[j] == i)[0][0]  # The order of unit i in atom j
            #             value += Lambda * frac_j[j] * self.Q[k] * np.prod([rho_total[i] for i in pre_list[j] if i < k]) / Mu
            #     rho_i_[i] = (1 - ((1 - rho_i) * alpha)[i]) * (1 - 1 / value)

            # Step 2: Normalize.
            if normalize:
                Gamma = rho_i_.sum() / (r * N)
                rho_i_ /= Gamma

            # Step 3: Convergence Test
            if abs(rho_i_ - rho_i).max() < epsilon:
                print ('Program stop in',n,'iterations in ', (time.time() - start_time), 'secs')
                self.rho_approx = rho_i_
                return rho_i_
            else:  # go to next step
                rho_i = np.array(rho_i_)
                rho_i_ = np.zeros(N)

    def Two_State_Approx_Mu_nj(self, epsilon=0.0001):  # This function includes heterogeneous mu
        """
        Similar to Two_State_Approx(), but mu is heterogeneous among units and atoms.
        We only consider no normalize and use effective lambda cases
        :param epsilon:
        :return: approximated utilization
        """
        keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 'pre_list', 'Mu_mat']
        N, K, Lambda, Mu, frac_j, pre_list, Mu_mat = [self.data_dict.get(key) for key in keys]
        try:
            alpha = self.alpha
        except:
            #print('Two state!')
            alpha = 0

        # Step 0: Initialization
        self.Cal_Q()  # This calculates Q, r, and G
        r = self.r  # average fraction of busy time for each unit in the system. Calculated in Cal_Q
        P_n = self.Cal_P_n()
        rho_i = np.full(N, r)  # utilization of each unit i. Initialization
        rho_i_ = np.zeros(N)  # temporary utilizations to store new value at each step
        n = 0
        use_effective_lambda = True
        # Step 1: Iteration
        start_time = time.time()
        while True:
            n += 1  # increase step by 1
            rho_total = (rho_i + (1 - rho_i) * alpha)  # rho_i denotes rho^{y}, rho_total denotes the total rho
            ######################
            # Use the effective lambda to get the most accurate P_n and Q for each iteration (This helps a lot)
            if use_effective_lambda:
                L = rho_total.sum()
                Lambda_eff = Get_Effective_Lambda(L, Mu, N)
                P_n = ErlangLoss(Lambda_eff, Mu, N)
                self.P_n = P_n
                self.Cal_Q(P_n=P_n, r=L/N)  # when use this Q, the old Q is overwritten and does not work
            ######################
            for i in range(N):  # for each unit
                value = 1  #
                for k in range(N):  # for each order
                    prod_g_j = 0  # Product term for each sum term
                    for j in self.G[i][k]:
                        prod_g_j += Lambda * frac_j[j] * self.Q[k] * np.prod(
                            rho_total[pre_list[j, :k]]) / Mu_mat[j][i]  # if alpha = 0, this is just rho in paranthasis
                    value += prod_g_j  # This line is edited for heterogeneous Mu
                rho_i_[i] = (1 - ((1 - rho_i) * alpha)[i]) * (
                        1 - 1 / value)  # again when alpha = 0, this is 1 in the paranthesis

            # Step 2: Convergence Test
            if abs(rho_i_ - rho_i).max() < epsilon:
                print ('Program stop in',n,'iterations in ', (time.time() - start_time), 'secs')
                # print(rho_i_)
                self.rho_approx = rho_i_
                return rho_i_
            else:  # go to next step
                rho_i = np.array(rho_i_)
                rho_i_ = np.zeros(N)

    def Get_MRT_Approx(self):  # Method 1 of getting response time as in Larson
        """
        Get mean response time based on approximation method
        :return:
        """
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try:  # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except:  # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx

        if self.P_n is None:
            P_n = self.Cal_P_n()
        else:
            P_n = self.P_n
        self.Cal_Q(r=np.sum(rho)/N, P_n=P_n)
        Q = self.Q
        # This is the fraction of dispatching unit i to atom j among all dispatches
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j, pre_j[n]] = Q[n] * np.prod(rho[pre_j[:n]]) * (1 - rho[pre_j[n]])
            q_nj[j, :] /= q_nj[j, :].sum()  # normalization
            q_nj[j, :] *= frac_j[j] * (1-P_n[-1])

        q_nj /= q_nj.sum()
        self.q_nj = q_nj  # store these values in the class
        MRT_j = np.sum(q_nj * t_mat, axis=1) / np.sum(q_nj, axis=1)
        MRT = np.sum(q_nj * t_mat)
        return MRT, MRT_j

def SumOfProduct(arr, k):  # calculates the sum product of all combanitions in arr given size k
    n = len(arr)
    # Initialising all the values to 0
    dp = [[0 for x in range(n + 1)] for y in range(n + 1)]
    # To store the answer for
    # current value of k
    cur_sum = 0
    # For k = 1, the answer will simply
    # be the sum of all the elements
    for i in range(1, n + 1):
        dp[1][i] = arr[i - 1]
        cur_sum += arr[i - 1]
        # Filling the table in bottom up manner
    for i in range(2, k + 1):
        # To store the elements of the current
        # row so that we will be able to use this sum
        # for subsequent values of k
        temp_sum = 0
        for j in range(1, n + 1):
            # We will subtract previously computed value
            # so as to get the sum of elements from j + 1
            # to n in the (i - 1)th row
            cur_sum -= dp[i - 1][j]

            dp[i][j] = arr[j - 1] * cur_sum
            temp_sum += dp[i][j]
        cur_sum = temp_sum
    sumprod_vec = np.array(dp).sum(axis=1)
    return sumprod_vec

def Get_Random_Sample(Lambda, type="exp"):
    if type == "exp":
        return np.random.exponential(1 / Lambda)
    elif type == "gamma":
        return np.random.gamma(shape=1, scale=1/Lambda)  # alpha=1, beta=1/lambda
    elif type == "lognormal":
        return np.random.lognormal(mean=(-np.log(Lambda*math.sqrt(2))), sigma=math.sqrt(np.log(2)))
    elif type == "weibull":
        return np.random.weibull(1) / Lambda
    else:
        return 0

class Three_State_Hypercube():
    def __init__(self, data_dict=None):
        # initilize data stored in self.data_dict
        self.keys_1 = ['N', 'N_1', 'N_2', 'K', 'Lambda_1', 'Mu_1', 'frac_j_1', 't_mat_1', 'pre_list_1']
        self.keys_2 = ['N', 'N_1', 'N_2', 'K', 'Lambda_2', 'Mu_2', 'frac_j_2', 't_mat_2', 'pre_list_2']
        self.data_dict_1 = dict.fromkeys(self.keys_1, None)
        self.data_dict_2 = dict.fromkeys(self.keys_2, None)
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys_1:
                    self.data_dict_1[k] = v
                if k in self.keys_2:
                    self.data_dict_2[k] = v
        self.time_exact = None
        self.time_alphahypercube = None  # MRT obtained from alpha hypercube
        self.time_linearalpha = None  # MRT obtained from linear alpha
        self.prob_dist_3state = None  # steadt state distribution when solved exactly by 3state hypercube
        self.pol_sub1 = None  # policy for service 1 for exact 3 state system
        self.pol_sub2 = None  # policy for service 2 for exact 3 state system
        self.rho_hyper_1 = None
        self.rho_hyper_2 = None

    def Update_Parameters(self, **kwargs):
        # update any parameters passed through kwargs
        for k, v in kwargs.items():
            if k in self.keys_1:
                self.data_dict_1[k] = v
            if k in self.keys_2:
                self.data_dict_2[k] = v


    def Update_alpha(self, method, subsystem):
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        N_sub1, N_sub2 = N - N_2, N - N_1
        # Specify method
        if method in 'exact':
            rho_sub1, rho_sub2 = self.sub1.rho_hyper, self.sub2.rho_hyper
        elif method in 'approximation':
            rho_sub1, rho_sub2 = self.sub1.rho_approx, self.sub2.rho_approx
        else:
            print('Wrong method')
        # Specify subsystem
        if subsystem == 1:
            rho = rho_sub1
            alpha = self.sub1.alpha
            N_sub_o = N_sub2
            N_me = N_1
            N_o = N_2
        elif subsystem == 2:
            rho = rho_sub2
            alpha = self.sub2.alpha
            N_sub_o = N_sub1
            N_me = N_2
            N_o = N_1
        else:
            print('Wrong subsystem!')
        alpha_ = np.array([0] * N_o + [
            rho[n + N_me - N_o] / (rho[n + N_me - N_o] + (1 - rho[n + N_me - N_o]) * (1 - alpha[n + N_me - N_o])) for n
            in range(N_o, N_sub_o)])
        return alpha_

    def Solve_3state_Hypercube(self):
        """
        Solve 3-state system with hypercube method for homogeneous mu,
        serving as the exact benchmark.
        :return: utilization in each subsystem
        """
        # Parameters
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu', 'frac_j', 'pre_list']
        Lambda_1, Mu_1, frac_j_1, pre_list_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        Lambda_2, Mu_2, frac_j_2, pre_list_2 = [self.sub2.data_dict.get(key) for key in keys_sub]

        if self.sub1.data_dict['Mu_vec'] is not None:
            print("Hete Mu!")
            flag_diff_mu = 1
            Mu_vec_1 = self.sub1.data_dict['Mu_vec']
            Mu_vec_2 = self.sub2.data_dict['Mu_vec']
        else:
            print("Homo Mu!")
            flag_diff_mu = 0


        pre_list_1_3state = pre_list_1.copy()  # make a copy of the preference list so that the original will not be modified
        pre_list_1_3state[pre_list_1_3state >= N_1] += N_2
        pre_list_2_3state = pre_list_2 + N_1

        start_time = time.time()  # staring time
        # Initialize States for each subsystem
        Num_State = 2 ** (N_1 + N_2) * 3 ** (N - N_1 - N_2)
        pol_sub1, pol_sub2 = np.ones([Num_State, K], dtype=int) * -1, np.ones([Num_State, K], dtype=int) * -1

        statusmat_sep = [np.base_repr(i, base=2)[1:] for i in range(2 ** (N_1 + N_2), 2 ** (
                N_1 + N_2 + 1))]  # Pad a 1 in front to make the length and take out the 1
        statusmat_joint = [np.base_repr(i, base=3)[1:] for i in range(3 ** (N - N_1 - N_2), 2 * 3 ** (
                N - N_1 - N_2))]  # Pad a 1 in front to make the length and take out the 1
        statusmat = [y + x for y in statusmat_joint for x in statusmat_sep]  # Get all the transitions
        transition = np.zeros([Num_State, Num_State])  # Initialize the transition matrix

        # Update upward transition rate
        for j in range(K):  # Loop through every atom
            for n in range(Num_State):  # Loop through states
                B_n = statusmat[n]
                # For EMS
                for i in pre_list_1_3state[j]:  # find the one in the preference to add the rate to the state
                    # Traverse all units in the sequence of pre_list of atom j in status B_n
                    if B_n[N - 1 - i] == '0':  # find the first available unit N-1-i shows unit i in the binary representation
                        pol_sub1[n, j] = i  # assign to policy
                        if i < N_1 + N_2:  # if it is a separate unit
                            m = n + 2 ** i  # the state transit to
                        else:  # if it is a joint unit
                            m = n + 2 ** (N_1 + N_2) * 3 ** (i - N_1 - N_2)  # the state transit to
                        transition[m, n] += Lambda_1 * frac_j_1[j]
                        break

                # For fire
                for i in pre_list_2_3state[j]:  # find the one in the preference to add the rate to the state
                    if B_n[N - 1 - i] == '0':  # find the first available unit
                        pol_sub2[n, j] = i  # assign to policy
                        if i < N_1 + N_2:  # if it is a separate unit
                            m = n + 2 ** i  # the state transit to
                        else:  # if it is a joint unit
                            m = n + 2 ** (N_1 + N_2) * 2 * 3 ** (i - N_1 - N_2)  # the state transit to
                        transition[m, n] += Lambda_2 * frac_j_2[j]
                        break

        # Update downward transition rate
        for n in range(Num_State):  # Loop through states
            B_n = statusmat[n]
            for i in range(N_1):  # Loop through every separate type-1
                if B_n[N - 1 - i] == '1':
                    m = n - 2 ** i
                    if flag_diff_mu == 1:
                        transition[m, n] = Mu_vec_1[i]
                    else:
                        transition[m, n] = Mu_1
            for i in range(N_1, N_1 + N_2):  # Loop through every separate type-2
                if B_n[N - 1 - i] == '1':
                    m = n - 2 ** i
                    if flag_diff_mu == 1:
                        transition[m, n] = Mu_vec_2[i - N_1]
                    else:
                        transition[m, n] = Mu_2
            for i in range(N_1 + N_2, N):  # Loop through every joint
                if B_n[N - 1 - i] == '1':  # for type-1
                    m = n - 2 ** (N_1 + N_2) * 3 ** (i - N_1 - N_2)
                    if flag_diff_mu == 1:
                        transition[m, n] = Mu_vec_1[i - N_2]
                    else:
                        transition[m, n] = Mu_1
                elif B_n[N - 1 - i] == '2':  # for type-2
                    m = n - 2 ** (N_1 + N_2) * 2 * 3 ** (i - N_1 - N_2)
                    if flag_diff_mu == 1:
                        transition[m, n] = Mu_vec_2[i - N_1]
                    else:
                        transition[m, n] = Mu_2

        # pol_sub1 and pol_sub2 directly give the preferred unit for each atom at every state (shape=(Num_State, K))
        self.pol_sub1 = pol_sub1
        self.pol_sub2 = pol_sub2
        # Set diagonal
        diag = np.diag(transition.sum(axis=0))
        transition -= diag
        # Solve for the steady state equation
        transition[-1] = np.ones(Num_State)
        b = np.zeros(Num_State)
        b[-1] = 1

        transition_sparse = sparse.csc_matrix(transition)
        prob_dist = spsolve(transition_sparse, b)

        self.prob_dist_3state = prob_dist
        print("------ %s seconds ------" % (time.time() - start_time))
        total_time = time.time() - start_time
        self.time_exact = total_time
        # Get rho
        rho_1, rho_2 = np.zeros(N - N_2), np.zeros(N - N_1)  # initialize
        for n in range(Num_State):
            B_n = statusmat[n]
            for i in range(N_1):  # separate type-1
                if B_n[N - 1 - i] == '1':
                    rho_1[i] += prob_dist[n]
            for i in range(N_1, N_1 + N_2):  # separate type-2
                if B_n[N - 1 - i] == '1':
                    rho_2[i - N_1] += prob_dist[n]
            for i in range(N_1 + N_2, N):  # joint unit
                if B_n[N - 1 - i] == '1':
                    rho_1[i - N_2] += prob_dist[n]  # serve type-1
                elif B_n[N - 1 - i] == '2':  # serve type-2
                    rho_2[i - N_1] += prob_dist[n]
        self.rho_hyper_1 = rho_1
        self.rho_hyper_2 = rho_2
        return rho_1, rho_2

    def Get_MRT_3state(self):
        """
        Calculate mean response time based on hypercube method
        :return: mean response time for subsystem 1 and 2
        """
        # Parameters
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['frac_j', 't_mat']
        frac_j_1, t_mat_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        frac_j_2, t_mat_2 = [self.sub2.data_dict.get(key) for key in keys_sub]
        N_sub1, N_sub2 = N - N_2, N - N_1

        not_all_busy_states_1 = np.unique(np.where(self.pol_sub1 != -1)[0])
        not_all_busy_states_2 = np.unique(np.where(self.pol_sub2 != -1)[0])

        q_nj_1 = np.zeros([K, N_sub1])

        list_sub1 = np.arange(N_sub1)
        add_ind = np.zeros(N_sub1, dtype=int)
        add_ind[N_1:] = N_2
        list_sub1 = list_sub1 + add_ind
        for n in np.arange(N_sub1):  # The last state has value 0
            # here we don't need last state so take :-1
            q_nj_1[:, n] = frac_j_1 * np.dot(self.prob_dist_3state[not_all_busy_states_1],
                                             self.pol_sub1[not_all_busy_states_1, :] == list_sub1[n])
        q_nj_1 /= q_nj_1.sum()
        MRT_1 = np.sum(q_nj_1 * t_mat_1)

        q_nj_2 = np.zeros([K, N_sub2])
        list_sub2 = np.arange(N_sub2) + N_1
        for n in np.arange(N_sub2):  # The last state has value 0
            q_nj_2[:, n] = frac_j_2 * np.dot(self.prob_dist_3state[not_all_busy_states_2],
                                             self.pol_sub2[not_all_busy_states_2, :] == list_sub2[
                                                 n])  # here we don't need last state so take :-1
        q_nj_2 /= q_nj_2.sum()
        MRT_2 = np.sum(q_nj_2 * t_mat_2)
        return MRT_1, MRT_2

    def Creat_Two_Subsystems(self):
        """
        Creating two subsystem to use linear alpha method
        """
        self.sub1 = self.Subsystem(
            dict((key[:-2], value) for (key, value) in self.data_dict_1.items() if len(key) > 3))  # take _1 off
        self.sub2 = self.Subsystem(
            dict((key[:-2], value) for (key, value) in self.data_dict_2.items() if len(key) > 3))  # take _2 off
        N_sub1 = self.data_dict_1['N'] - self.data_dict_1['N_2']
        N_sub2 = self.data_dict_2['N'] - self.data_dict_2['N_1']
        K = self.data_dict_1['K']
        self.sub1.Update_Parameters(N=N_sub1, K=K)
        self.sub2.Update_Parameters(N=N_sub2, K=K)
        self.sub1.alpha = np.zeros(N_sub1)
        self.sub2.alpha = np.zeros(N_sub2)

    def Reset_Alpha(self):
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        N_sub1 = N - N_2
        N_sub2 = N - N_1
        self.sub1.alpha = np.zeros(N_sub1)
        self.sub2.alpha = np.zeros(N_sub2)

    def Linear_Alpha(self, use_effective_lambda=True, normalize=False, epsilon=0.0001):
        """
        Main algorithm for estimating utilization
        :param use_effective_lambda: if use effective lambda or not
        :param normalize: if normalize or not
        :param epsilon:
        :return: runtime
        """
        if normalize:  # if we normalize rho after each iteration. In this function, default is to normalize
            self.Cal_P_b()  # calculate the block probability, which in turn gives average utility. This also initializes the intial alpha that is fast for computation
        ite = 0
        run = True

        # Check how heterogeneous the mu is
        if self.sub1.data_dict['Mu_vec'] is not None:
            flag_diff_mu = 1
        elif self.sub1.data_dict['Mu_mat'] is not None:
            flag_diff_mu = 2
        else:
            flag_diff_mu = 0

        start_time = time.time()
        while run:
            ite += 1
            # subsystem 1
            if flag_diff_mu == 2:
                self.sub1.Two_State_Approx_Mu_nj()
            else:
                self.sub1.Two_State_Approx(use_effective_lambda=use_effective_lambda, normalize=normalize, flag_diff_mu=flag_diff_mu)
            alpha = self.Update_alpha(method='approx', subsystem=1)
            self.sub2.alpha = alpha

            # subsystem 2
            if flag_diff_mu == 2:
                self.sub2.Two_State_Approx_Mu_nj()
            else:
                self.sub2.Two_State_Approx(use_effective_lambda=use_effective_lambda, normalize=normalize, flag_diff_mu=flag_diff_mu)
            alpha = self.Update_alpha(method='approx', subsystem=2)
            if (max(abs(alpha - self.sub1.alpha))< epsilon):
                run = False
            self.sub1.alpha = alpha
        # print("------ Linear Alpha run %s seconds ------" % (time.time() - start_time))
        # print('Number of iteration:', ite)
        self.time_linearalpha = time.time() - start_time
        self.ite = ite
        return time.time() - start_time

    def Cal_P_b(self):  # Calculate the block probability of the two subsystems.
        # as a consequence, this might make linear-alpha slow. One alternative is to do the update as before without normalize like this
        # and then normalize after convergence.
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu']
        Lambda_1, Mu_1 = [self.sub1.data_dict.get(key) for key in keys_sub]
        Lambda_2, Mu_2 = [self.sub2.data_dict.get(key) for key in keys_sub]

        if N_1 == N_2 == 0:  # all cross-trained
            P_b = ErlangLoss(Lambda_1 + Lambda_2, (Lambda_1 + Lambda_2) / (Lambda_1 / Mu_1 + Lambda_2 / Mu_2), N)[
                -1]  # blocking probability
            self.sub1.P_b, self.sub2.P_b = P_b, P_b  # update the two blocking probabilities
            r_1, r_2 = Lambda_1 / (N * Mu_1) * (1 - P_b), Lambda_2 / (N * Mu_2) * (
                    1 - P_b)  # This is the average utilizations of each service
            self.sub1.alpha = np.ones(N) * r_2 / (1 - r_1)  # initialize alpha this way is faster
            self.sub2.alpha = np.ones(N) * r_1 / (1 - r_2)
        else:  # general case, This includes all cross-trained case but a little bit more complicated
            N_c = N - N_1 - N_2
            P_b1_1, P_b1_2 = P_b1(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N)
            P_b2_1, P_b2_2 = P_b2(Lambda_1, Lambda_2, Mu_1, Mu_2, N_1, N_2, N)
            P_b_1, P_b_2 = (P_b1_1 + P_b2_1) / 2, (P_b1_2 + P_b2_2) / 2
            self.sub1.P_b, self.sub2.P_b = P_b_1, P_b_2  # update the two blocking probabilities
            r_1, r_2 = Lambda_1 / ((N - N_2) * Mu_1) * (1 - P_b_1), Lambda_2 / ((N - N_1) * Mu_2) * (
                    1 - P_b_2)  # This is the average utilizations of each service
            if r_1 + r_2 > 1:  # This ensures that the intializes alpha to be less than 1
                r_1, r_2 = r_1 / (r_1 + r_2), r_2 // (r_1 + r_2)
            self.sub1.alpha = np.concatenate((np.zeros(N_1), np.ones(N_c) * r_2 / (
                    1 - r_1)))  # initialize alpha this way is faster. 0s for the separate units
            self.sub2.alpha = np.concatenate((np.zeros(N_1), np.ones(N_c) * r_1 / (1 - r_2)))

    def Get_MRT_Approx_3state(self):
        """
        Calculate mean response time based on approximation method
        :return: mean response time for subsystem 1 and 2
        """
        keys = ['N', 'N_1', 'N_2']
        N, N_1, N_2 = [self.data_dict_1.get(key) for key in keys]
        rho_1_approx = self.sub1.rho_approx
        rho_2_approx = self.sub2.rho_approx

        self.sub1.rho_total_approx = rho_1_approx + np.append(np.zeros(N_1), rho_2_approx[N_2:])
        self.sub2.rho_total_approx = rho_2_approx + np.append(np.zeros(N_2), rho_1_approx[N_1:])

        MRT_1, MRT_1_j = self.sub1.Get_MRT_Approx()
        MRT_2, MRT_2_j = self.sub2.Get_MRT_Approx()
        return MRT_1, MRT_2, MRT_1_j, MRT_2_j

    def Simulator_Mu_nj(self, type, service_distribution="exp", seed=9001, time_horizon=500000):
        """
        Do simulation of heterogeneous mu
        :param type: "mat" for mu_ij; "vec" for mu_i (among units)
        :param seed: random seed
        :param time_horizon: time of simulation
        :return: MRT_1, MRT_2, rho_total/time_horizon
        """
        start_time = time.time()
        np.random.seed(seed)
        keys = ['N', 'N_1', 'N_2', 'K']
        N, N_1, N_2, K = [self.data_dict_1.get(key) for key in keys]
        keys_sub = ['Lambda', 'Mu', 'frac_j', 'pre_list', 't_mat']
        Lambda_1, Mu_1, frac_j_1, pre_list_1, t_mat_1 = [self.sub1.data_dict.get(key) for key in
                                                                   keys_sub]
        Lambda_2, Mu_2, frac_j_2, pre_list_2, t_mat_2 = [self.sub2.data_dict.get(key) for key in
                                                                   keys_sub]
        if type == "mat":
            Mu_mat_1, Mu_mat_2 = self.sub1.data_dict['Mu_mat'], self.sub2.data_dict['Mu_mat']
        elif type == "vec":
            Mu_vec_1, Mu_vec_2 = self.sub1.data_dict['Mu_vec'], self.sub2.data_dict['Mu_vec']
        # change preference list to incorporate separate units
        pre_list_1_simu = pre_list_1.copy()  # make a copy of the preference list so that the original will not be modified
        pre_list_1_simu[pre_list_1_simu >= N_1] += N_2
        pre_list_2_simu = pre_list_2 + N_1

        # Generate the arrivals in each atom, stored as (arrival_time, atom_number, arrival_type)
        arrival_event = []
        for j in range(K):
            # for each atom
            Lambda = Lambda_1 * frac_j_1[j] + Lambda_2 * frac_j_2[j]  # Total arrival rate in atom j
            p = np.array([Lambda_1 * frac_j_1[j], Lambda_2 * frac_j_2[j]]) / Lambda
            cumulative_time = 0
            while cumulative_time < time_horizon:
                cumulative_time += np.random.exponential(1 / Lambda)
                arrival_type = np.random.random()
                if arrival_type < p[0]:
                    arrival_event.append((cumulative_time, j, 1))
                else:
                    arrival_event.append((cumulative_time, j, 2))
        arrival_event = sorted(arrival_event, key=lambda x: x[0], reverse=False)  # Sort all arrivals
        arrival_event = [i for i in arrival_event if
                         i[0] <= time_horizon]  # Delete those arrive after the end of simulation

        rho_sub1 = np.zeros(N)
        rho_sub2 = np.zeros(N)
        rho_total = np.zeros(N)
        q_nj_1 = np.zeros([K, N])
        q_nj_2 = np.zeros([K, N])
        q1 = 0
        q2 = 0
        busy_unit = []  # every element is a tuple (unit_i, service_end_time, service_time, call_type)
        for arrival in arrival_event:
            # For all arrivals, where arrival = (arrival_time, atom_j, call_type)
            # e.g. arrival = (114.514, 13, 2)

            # 1. Find all units that finished call before this arrival and after former arrival
            finished_units = [busy for busy in busy_unit if busy[1] <= arrival[0]]
            for finish in finished_units:
                rho_total[finish[0]] += finish[2]
                if finish[3] == 1:
                    rho_sub1[finish[0]] += finish[2]
                elif finish[3] == 2:
                    rho_sub2[finish[0]] += finish[2]

            # 2. End services and update free_units
            # Delete all units that end the service before this arrival and after former arrival
            busy_unit = [busy for busy in busy_unit if busy[1] > arrival[0]]
            # Update current free units
            free_unit = [i for i in range(N) if all(i != busy[0] for busy in busy_unit)]

            # 3. Respond to new arrival call
            # Get preference list
            if arrival[2] == 1:
                q1 += 1
                pre = pre_list_1_simu[arrival[1]]
            else:
                q2 += 1
                pre = pre_list_2_simu[arrival[1]]
            for unit in pre:
                if unit in free_unit:
                    # find the first free unit in prefer list
                    if arrival[2] == 1:
                        if unit < N_1 + N_2:
                            u = unit  # Separate
                        else:
                            u = unit - N_2  # joint
                        if type == "mat":
                            service_time = Get_Random_Sample(Mu_mat_1[arrival[1]][u], service_distribution)
                        elif type == "vec":
                            service_time = Get_Random_Sample(Mu_vec_1[u], service_distribution)
                        else:
                            service_time = Get_Random_Sample(Mu_1, service_distribution)
                        q_nj_1[arrival[1]][unit] += 1
                    else:
                        if type == "mat":
                            service_time = Get_Random_Sample(Mu_mat_2[arrival[1]][unit - N_1], service_distribution)
                        elif type == "vec":
                            service_time = Get_Random_Sample(Mu_vec_2[unit - N_1], service_distribution)
                        else:
                            service_time = Get_Random_Sample(Mu_2, service_distribution)
                        q_nj_2[arrival[1]][unit] += 1
                    service_end_time = service_time + arrival[0]
                    busy_unit.append((unit, service_end_time, service_time, arrival[2]))
                    break
        print("######## Simulate ########")
        print("Run for", time.time() - start_time, "seconds")
        rho_sim_1 = [i for i in rho_sub1 / time_horizon if i != 0]
        rho_sim_2 = [i for i in rho_sub2 / time_horizon if i != 0]
        q_nj_1 = np.hstack((q_nj_1[:, :N_1], q_nj_1[:, N_1 + N_2:]))
        q_nj_1 /= np.sum(q_nj_1)
        q_nj_2 = q_nj_2[:, N_1:]
        q_nj_2 /= np.sum(q_nj_2)
        MRT_1 = np.sum(q_nj_1 * t_mat_1)
        MRT_2 = np.sum(q_nj_2 * t_mat_2)
        return MRT_1, MRT_2, rho_total/time_horizon, rho_sim_1, rho_sim_2, time.time()-start_time

    class Subsystem(
        Two_State_Hypercube):  # This class belongs to the three-state class and is a children class of Two_state_hyper
        def __init__(self, data_dict=None):
            super().__init__(data_dict=data_dict)
            self.alpha = None  # initilize alpha to be none. The alpha value for this subsystem. This alpha is intialized in the Cal_P_b function.
            self.rho_total_approx = None  # total rho for this subsystem when calculated by approximation. rho_1+rho_2 for joint

        def Cal_Trans(self):  # This contains alpha. Overwrites the original function in parental class
            # This is not the most efficient. Good enough for now.
            keys = ['N', 'K', 'Lambda', 'Mu', 'pre_list', 'frac_j']
            N, K, Lambda, Mu, pre_list, frac_j = [self.data_dict.get(key) for key in keys]
            alpha = self.alpha

            Num_state = 2 ** N
            A = np.zeros([Num_state, Num_state])  # Initilize

            # Calculate upward transition
            for s in range(Num_state - 1):
                for j in range(K):
                    pre_list_j = pre_list[j]  # pre_list for atom j
                    alpha_prod = 1
                    for k in range(N):  # find states in which kth preferred unit is free
                        unit = pre_list_j[k]
                        if not s & (1 << unit):  # 1 << unit <=> 2**unit, a bitwise operation
                            # if it is free
                            s_ = s ^ (1 << unit)  # The state it transitions to
                            A[s, s_] += Lambda * frac_j[j] * alpha_prod * (1 - alpha[unit])
                            alpha_prod *= alpha[unit]
                            A[s_, s] = Mu
            return A

        def Cal_P_n(self):  # this overwrites the P_n in the parental class
            '''
                :Lambda_v, Mu_v, alpha: Inputs
                "Output: P_n
                This one assumes every combination is with equal probability. Getting the steady state probability P_n
            '''
            keys = ['N', 'Lambda', 'Mu']
            N, Lambda, Mu = [self.data_dict.get(key) for key in keys]

            Lambda_BD = np.ones(N) * Lambda  # Lambdas of the birth and death chain
            sumprod_vec = SumOfProduct(self.alpha,
                                       N)  # this calculates all the combinations of sum of product of alpha's in one run. Significantly reduce TIME.
            for i in range(N):
                num_comb = comb(N, i + 1)  # number of totoal combinations when in total i+1 units busy
                Lambda_BD[N - i - 1] = Lambda / num_comb * (num_comb - sumprod_vec[
                    i + 1])  # Using the equal probability assmption, this is equation lambda(k) in Thm 1
                # When there is no much difference of lambdas, we just assume it is Lambda rather than do massive computation
                if Lambda - Lambda_BD[N - i - 1] < 0.001:
                    break
            # print('Lambda_BD',Lambda_BD)
            Mu_BD = Mu * (np.array(range(N)) + 1)
            P_n = ErlangLoss(Lambda_BD, Mu_BD)
            return P_n

        def Get_MRT_Hypercube(
                self):  # overwrites the one for 2-class cases. Much more complicated because cannot directly use pol
            keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
            N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
            alpha = self.alpha
            prob_dist = self.prob_dist
            Num_state = 2 ** N

            q_nj = np.zeros([K, N])  # probability of assigning unit i to node j
            for s in range(Num_state - 1):
                for j in range(K):
                    pre_list_j = pre_list[j]
                    alpha_prod = 1
                    # i = 0
                    for k in range(N):  # find states in which kth preferred unit is free
                        unit = pre_list_j[k]
                        if not s & (1 << unit):  # if it is free
                            q_nj[j, unit] += frac_j[j] * alpha_prod * (1 - alpha[unit]) * prob_dist[s]
                            alpha_prod *= alpha[
                                unit]  # this is to capture states that has 1 there so it will not be dispatched
                            # i += 1
            q_nj /= q_nj.sum()  # this is the same as divide by (1-P_allbusy)
            MRT = np.sum(q_nj * t_mat)
            MRT_j = np.sum(q_nj * t_mat, axis=1) / np.sum(q_nj, axis=1)
            return MRT, MRT_j


system = Three_State_Hypercube({'Lambda_1': 10, 'Mu_1': 20, 'Lambda_2': 10, 'Mu_2': 20})
system.Update_Parameters(N=7, N_1=2, N_2=3, K=30)

system.Creat_Two_Subsystems()
system.sub1.Random_Mu(20, radius=0.8)
system.sub1.Random_Pref(seed=1)
system.sub1.Random_Time_Mat(t_min=1, t_max=10, seed=1)
system.sub1.Random_Fraction(seed=1)

system.sub2.Random_Mu(20, radius=0.8)
system.sub2.Random_Pref(seed=1)
system.sub2.Random_Time_Mat(t_min=1, t_max=10, seed=1)
system.sub2.Random_Fraction(seed=1)

system.Solve_3state_Hypercube()
print(system.rho_hyper_1, system.rho_hyper_2, system.Get_MRT_3state())

system.Reset_Alpha()
system.Linear_Alpha()
print(system.sub1.rho_approx, system.sub2.rho_approx, system.Get_MRT_Approx_3state())

MRT_1, MRT_2, _, rho_sim_1, rho_sim_2, _ = system.Simulator_Mu_nj(type="vec")
print(rho_sim_1, rho_sim_2, MRT_1, MRT_2)
