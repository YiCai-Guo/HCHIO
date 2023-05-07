import random
from copy import deepcopy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import os

plt.rcParams['font.sans-serif'] = ['STSong']
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class MY_CHIO:
    def __init__(self, fjsp, epoch=10000, pop_size=100, brr=0.15, max_age=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            brr (float): Basic reproduction rate, default=0.15
            max_age (int): Maximum infected cases age, default=10
        """
        # super().__init__(**kwargs)  # super() 用来调用父类(基类)的方法  # 这里kwargs没有值，用**拆解之后依然为空，所以就是没有传入任何参数
        self.epoch = epoch
        self.pop_size = pop_size
        self.brr = brr
        self.max_age = max_age
        self.env = fjsp
        self.one_dims = len(fjsp.job_repeat)
        self.pop = []
        self.FIT_ID = 1  # 一个个体的适应度值的位置索引
        self.POS_ID = 0  # 一个个体的位置向量索引
        self.list_global_best = []
        self.last_best = float('inf')
        self.first_best_epoch = float('-inf')
        self.good_pop = []

    def solve(self):
        """
        主函数循环体，控制算法总流程，实验结果记录
        """
        self.pop = self.create_population()
        self.initialize_variables()
        for gen in range(0, self.epoch):
            time_epoch = time.perf_counter()
            self.evolve(gen)
            pop_sort, g_best = self.global_best_record(self.pop, gen)
            time_epoch = time.perf_counter() - time_epoch
            print(f"DataSet: {self.env.dataset_name}, Epoch: {gen}, Current best: {pop_sort[0][self.FIT_ID]}, "
                  f"Global best: {self.list_global_best[-1][self.FIT_ID]}, Runtime: {time_epoch:.5f} seconds")
        # print(self.pop)
        return self.list_global_best[-1][self.POS_ID], self.list_global_best[-1][self.FIT_ID], self.first_best_epoch

    def global_best_record(self, pop, it, save=True):
        """
        Update global best and current best solutions in history object.
        Also update global worst and current worst solutions in history object.
        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current/global best to history,
                        False if you just want to update current/global best
            it (int): 当前代数，用于记录哪个回合收敛
        Returns:
            list: Sorted population and the global best solution
        """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.FIT_ID])
        current_best = sorted_pop[0]
        current_worst = sorted_pop[-1]
        if len(self.list_global_best) == 0:
            self.list_global_best.append(current_best)  # 记录每个回合的全局最优解
            self.good_pop.append(current_best)          # 记录优秀种群
            self.first_best_epoch = it
        else:
            if current_best[self.FIT_ID] < self.good_pop[-1][self.FIT_ID]:
                self.first_best_epoch = it
            if current_best[self.FIT_ID] <= self.good_pop[-1][self.FIT_ID]:
                if not np.array_equal(current_best[self.POS_ID], self.good_pop[-1][self.POS_ID]):
                    self.good_pop.append(current_best)
                self.list_global_best.append(current_best)
            else:
                global_last = deepcopy(self.list_global_best[-1])
                self.list_global_best.append(global_last)
        self.last_best = self.good_pop[-1][self.FIT_ID]
        return deepcopy(sorted_pop), deepcopy(current_best)

    def order_cross_1(self, p1, p2, r1):
        """
        POX交叉操作：
        对p1执行r1选择，即r1集合内工件的对应的工件加工顺序保持不变，然后产生c1
        Args：
            p1： np.array类型，位置向量，两段式编码的第一段，长度为 self.one_dims
            p2： np.array类型，位置向量，两段式编码的第一段，长度为 self.one_dims
            r1: 工件集合，全部工件集合的半集，与r2构成工件全集
        Return：
            c1，c2：np.array类型，交叉后的order段的编码，长度为 self.one_dims
        """
        c1, c2 = [], []
        j = 0
        for i in range(self.one_dims):
            if p1[i] in r1:
                c1.append(p1[i])
            else:
                while p2[j] in r1:
                    c2.append(p2[j])
                    j += 1
                c2.append(p1[i])
                c1.append(p2[j])
                j += 1
        while j != self.one_dims:
            c2.append(p2[j])
            j += 1
        return np.array(c1), np.array(c2)

    def code_separate(self, code):
        """
        将两段式编码code转化为：job_sequence（前半段编码）、机器选择列表
        Args：
            code：np.array类型，两段式编码，前半段为job排序，后半段为机器选择结果，np.array类型，长度为 2*self.one_dims
        Returns:
            job_sequence：np.array类型，code的前半段编码，长度为 self.one_dims
            machine_separate：list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
        """
        machine_separate = []
        for i in range(self.env.job_num):
            machine_separate.append([])
        for j in range(self.one_dims):
            job = code[j] - 1
            machine_separate[job].append(code[self.one_dims + j])
        job_sequence = deepcopy(code[0:self.one_dims])
        return job_sequence, machine_separate

    def code_mix(self, job_part, machine_list):
        """
        将两部分结果混合，构成两段式编码
        Args：
            job_part: np.array类型，长度 self.one_dims
            machine_list: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
        Returns:
            code: np.array类型，两段式编码，长度为 2*self.one_dims
        """
        op_record = np.zeros(self.env.job_num, dtype=int)  # 用于记录已经加工到该工件的第几个工序
        machine_record = []
        for i in range(self.one_dims):
            job = job_part[i] - 1
            op = op_record[job]
            machine_record.append(machine_list[job][op])
            op_record[job] += 1
        machine_part = np.array(machine_record)
        code = np.hstack((job_part, machine_part))
        return code

    def machine_mpx(self, machine_1, machine_2):
        """
        将两机器选择结果，按照50%概率执行交叉操作
        Args：
            machine_list_1: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
            machine_list_2: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
        Returns:
            machine_list_1: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
            machine_list_2: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
        """
        for j in range(self.env.job_num):
            op_n = len(machine_1[j])
            for k in range(op_n):
                if machine_1[j][k] != machine_2[j][k]:
                    if np.random.randint(0, 2):
                        machine_1[j][k], machine_2[j][k] = machine_2[j][k], machine_1[j][k]
        return machine_1, machine_2

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        r = {i+1 for i in range(self.env.job_num)}  # 工序编号全集 从1开始计数编号
        for i in range(0, self.pop_size):
            temp = self.pop[i]
            j1, m1 = self.code_separate(temp[self.POS_ID])
            if np.random.uniform() <= 0.8:
                if np.random.uniform() <= self.brr:
                    rand = self.immunity_type_list[i]
                    # if rand < (1.0 / 3) * self.brr:
                    if rand == 1:
                        idx_candidates = np.where(self.immunity_type_list == 1)  # Infected list
                        idx_selected = np.random.choice(idx_candidates[0])
                        # 轮盘赌法
                        # fit_list = np.array([self.pop[item][self.FIT_ID] for item in idx_candidates[0]])
                        # s = fit_list.sum()
                        # idx_selected = np.random.choice(idx_candidates[0], p=fit_list / s)
                        # 头狼引领法
                        # the_min = np.where(fit_list == min(fit_list))
                        # idx_selected = idx_candidates[0][np.random.choice(the_min[0])]  # Found the index of min
                        r1 = set(random.sample(r, int(self.env.job_num / 2)))  # 工序编号的半集1
                        # is_corona_list[i] = True
                    elif rand == 0:
                        idx_candidates = np.where(self.immunity_type_list == 0)  # Susceptible list
                        idx_selected = np.random.choice(idx_candidates[0])
                        # 轮盘赌法
                        # fit_list = np.array([self.pop[item][self.FIT_ID] for item in idx_candidates[0]])
                        # s = fit_list.sum()
                        # idx_selected = np.random.choice(idx_candidates[0], p=fit_list / s)
                        # 头狼引领法
                        # the_min = np.where(fit_list == min(fit_list))
                        # idx_selected = idx_candidates[0][np.random.choice(the_min[0])]   # Found the index of min
                        r1 = set(random.sample(r, int(self.env.job_num / 3)))
                    else:  # rand == 2
                        idx_candidates = np.where(self.immunity_type_list == 2)  # Immunity list
                        idx_selected = np.random.choice(idx_candidates[0])
                        # 轮盘赌法
                        # fit_list = np.array([self.pop[item][self.FIT_ID] for item in idx_candidates[0]])
                        # s = fit_list.sum()
                        # idx_selected = np.random.choice(idx_candidates[0], p=fit_list / s)
                        # 头狼引领法
                        # the_min = np.where(fit_list == min(fit_list))
                        # idx_selected = idx_candidates[0][np.random.choice(the_min[0])]  # # Found the index of min
                        r1 = set(random.sample(r, int(self.env.job_num / 4)))
                else:
                    pop_num = {k for k in range(len(self.pop))} - {i}
                    idx_selected = np.random.choice(list(pop_num))
                    # idx_selected = self.choose_approximate_fitness(temp[self.FIT_ID])
                    size = np.random.randint(2, self.env.job_num-1)
                    r1 = set(random.sample(r, size))  # 工序编号的半集1
                p2 = self.pop[idx_selected]
                j2, m2 = self.code_separate(p2[self.POS_ID])

                # order 交叉操作
                j1_1, j2_1 = self.order_cross_1(j1, j2, r1)
                c1_1, c2_1 = self.code_mix(j1_1, m1), self.code_mix(j2_1, m2)

                temp = self.choose_min(c1_1, temp)
                self.pop[idx_selected] = self.choose_min(p2, c2_1)

                # machine 交叉操作
                m1_1, m2_1 = self.machine_mpx(m1, m2)
                c1_2, c2_2 = self.code_mix(j1_1, m1_1), self.code_mix(j2_1, m2_1)

                temp = self.choose_min(c1_2, temp)
                self.pop[idx_selected] = self.choose_min(p2, c2_2)

            if np.random.uniform() < 0.3:
                jp, mp = self.code_separate(temp[self.POS_ID])
                temp1 = self.variation_insert(jp, 0.05+0.05*(1-((2.71**(epoch/self.epoch)-1)/1.71)**1.5))
                temp2 = self.variation_machine(mp, 0.05+0.05*(1-((2.71**(epoch/self.epoch)-1)/1.71)**1.5))
                te = self.code_mix(temp1, temp2)
                temp = deepcopy(self.choose_min(temp, te))

            # Step 3.5: variable_neighborhood_search
            temp_vns = self.variable_neighborhood_search(temp[0])
            t, _, _ = self.env.active_schedule(temp_vns)
            if t <= temp[1]:
                temp = deepcopy([temp_vns, t])

            # Step 4: Update herd immunity population
            if temp[self.FIT_ID] < self.pop[i][self.FIT_ID]:
                self.pop[i] = deepcopy(temp)
                self.age_list[i] = 0
            else:
                self.age_list[i] += 1

            # Step 5: Fatality condition
            if self.age_list[i] >= self.max_age:
                # if self.pop[i][self.FIT_ID] == self.last_best:
                #     self.age_list[i] = 0
                # else:
                # self.pop[i] = self.create_offspring()
                self.pop[i] = self.create_solution()
                self.immunity_type_list[i] = self.immunity_type_list[i]
                self.age_list[i] = 0

    def choose_approximate_fitness(self, f):
        """
        死亡个体的替代问题，随机选取，近期一定比例（proportion）全局最优解替代
        Args:
            f: float类型，适应度值
        Returns:
            idx_select: int类型，self.pop种群被选中个体的位置下标
        """
        fit_list = [item[self.FIT_ID] for item in self.pop]
        fit_sort = sorted(fit_list)
        b = fit_sort.index(f)
        if b == 0:
            idx_select = np.random.choice([i for i in range(int(0.1*len(self.pop))) ])
        else:
            res = fit_sort[b - 1]
            idx_candidates = np.where(np.array(fit_list) == res)  # Susceptible list
            idx_select = np.random.choice(idx_candidates[0])
        return idx_select

    def create_offspring(self, proportion=0):  # 后 proportion 的 self.good_pop 才有资格参与产生子代
        """
        死亡个体的替代问题，随机选取，近期一定比例（proportion）全局最优解替代
        Args:
            proportion: float类型，self.list_global_best 后半部分的比例的个体替代死亡个体
        Returns:
            [code: np.array类型, target: float类型，makespan]
        """
        parents = len(self.good_pop)
        which = np.random.choice([i for i in range(int(proportion * parents), parents)])
        last_best = self.good_pop[which][self.POS_ID]
        sequence_p, machine_p = self.code_separate(last_best)
        s_p = self.variation_reverse(sequence_p)
        m_p = self.variation_machine(machine_p)
        code = self.code_mix(s_p, m_p)
        target, _, _ = self.env.active_schedule(code)
        return [code, target]

    def variation_reverse(self, order, vc=0.5):
        """
        逆序变异，近期一定比例（proportion）全局最优解替代
        Args:
            order: np.array类型，self.list_global_best 后半部分的比例的个体替代死亡个体
            vc: float类型，order的 vc 比例长度执行逆序操作
        Returns:
            order_op: np.array类型
        """
        order_op = deepcopy(order)
        one = np.random.randint(0, self.one_dims)
        if one+int(self.one_dims*vc) < self.one_dims:
            end = one+int(self.one_dims*vc)
            while end != one and one < end:
                order_op[end], order_op[one] = order_op[one], order_op[end]
                end -= 1
                one += 1
            return order_op
        elif one-int(self.one_dims*vc) >= 0:
            start = one-int(self.one_dims*vc)
            while start != one and start < one:
                order_op[start], order_op[one] = order_op[one], order_op[start]
                start += 1
                one -= 1
            return order_op
        else:
            return order_op

    def variation_insert(self, order, vc=0.5):
        """
        工序插入变异，近期一定比例（proportion）全局最优解替代
        Args:
            order: np.array类型，长度为self.one_dims
            vc: float类型，order的 vc 比例个位点执行向前插入变异操作
        Returns:
            order_op: np.array类型，长度为self.one_dims
        """
        order_op = deepcopy(order)
        choices = [i for i in range(self.one_dims)]
        select = np.random.choice(choices, int(self.one_dims*vc))
        for i in select:
            search = i
            while order_op[search] == order_op[i] and search != 0:
                search -= 1
            temp = order_op[i]
            order_op = np.delete(order_op, i)
            order_op = np.insert(order_op, search, temp)
        return order_op

    def variation_machine(self, m1, vc=0.5, critical_machine=-1):
        """
        机器变异，按照 vc 概率去参加变异
        Args:
            m1: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
            vc: 变异的概率
            critical_machine: 本来打算传入关键机器，目前无作用
        Returns:
            m1: list类型，二维，形状为（工件数，该工件的工序数），[[job1的机器选择结果],[job2的机器选择结果],……]
        """
        for i in range(len(m1)):
            for j in range(len(m1[i])):
                if np.random.choice([0, 1], p=[1-vc, vc]):
                    m1[i][j] = np.random.choice(self.env.machine_sign[i][j])
                    if critical_machine >= 0:
                        if m1[i][j] == critical_machine:
                            idx = np.argmin(self.env.machine_sign[i][j])
                            m1[i][j] = self.env.machine_sign[i][j][idx]
        return m1

    def variable_neighborhood_search(self, code):
        """
        逆序变异，近期一定比例（proportion）全局最优解替代

        Args:
            code: np.array类型，长度 self.one_dims * 2
        Returns:
            temp: np.array类型，长度 self.one_dims * 2
        """
        temp = deepcopy(code)
        makespan, last_machine_list, machine_record = self.env.active_schedule(code)
        use_ratio = np.zeros(self.env.machine_num)
        for i in range(self.env.machine_num):
            count = 0
            for j in range(len(machine_record[i])):
                process = machine_record[i][j][2][1] - machine_record[i][j][2][0]
                count += process
            use_ratio[i] = count/makespan

        op_count = np.zeros(self.env.job_num, dtype=int)
        for i in range(self.one_dims):
            job = code[i] - 1
            op = op_count[job]
            machine = code[self.one_dims+i]
            if machine not in last_machine_list:
                op_count[job] += 1
                continue
            else:
                if np.random.choice([0, 1], p=[0.95, 0.05]):
                    optional_m_list = self.env.machine_sign[job][op]
                    optional_m_t = self.env.machine_time[job][op]
                    use_ratio_list = [use_ratio[i-1] for i in optional_m_list]
                    min_ratio_id = use_ratio_list.index(min(use_ratio_list))
                    min_time_id = optional_m_t.index(min(optional_m_t))
                    if machine == optional_m_list[min_time_id]:
                        temp[self.one_dims+i] = optional_m_list[min_ratio_id]
                    elif machine == optional_m_list[min_ratio_id]:
                        temp[self.one_dims+i] = optional_m_list[min_time_id]
                    else:
                        if np.random.randint(2):
                            temp[self.one_dims + i] = optional_m_list[min_ratio_id]
                        else:
                            temp[self.one_dims + i] = optional_m_list[min_time_id]
                op_count[job] += 1
        return temp

    def choose_min(self, code1, code2):
        if len(code1) != 2:
            t1, _, _ = self.env.active_schedule(code1)
            code1 = [code1, t1]
        if len(code2) != 2:
            t2, _, _ = self.env.active_schedule(code2)
            code2 = [code2, t2]
        if code1[self.FIT_ID] < code2[self.FIT_ID]:
            return code1
        else:
            return code2

    @staticmethod
    def cross(x, y):
        start = int(len(x)/2)
        for i in range(start, len(x)):
            if np.random.rand() > 0.5:
                x[i], y[i] = y[i], x[i]
        return x, y

    def sort(self, pop, immunity_type_list, age_list):
        pop1 = deepcopy(pop)
        immunity_type_list1 = deepcopy(immunity_type_list)
        age_list1 = deepcopy(age_list)
        fit_list = [agent[self.FIT_ID] for agent in pop]
        index = np.array(fit_list).argsort()
        for i in range(len(pop)):
            pop1[i] = pop[index[i]]
            immunity_type_list1[i] = immunity_type_list[index[i]]
            age_list1[i] = age_list[index[i]]
        return pop1, immunity_type_list1, age_list1

    def create_population(self, pop_size=None):  # 创建一个种群
        """
        Args:
            pop_size (int): number of solutions

        Returns:
            list: population or list of solutions/agents
        """
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        for i in range(0, pop_size):
            first = self.create_solution()
            pop.append(first)
        return pop  # pop是一个list，[position（np数组），[fit, obj]]

    def create_solution(self):
        """
        创建一个单独的解（个体）
        To get the position, target wrapper [fitness and obj list]
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [fitness, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: fitness
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        Returns:
            list: wrapper of solution with format [position, [fitness, [obj1, obj2, ...]]]
            # 创造一个解：[位置向量，优化目标]
        """
        rand_list = random.sample(range(4 * self.one_dims), self.one_dims)
        index = np.array(rand_list).argsort()  # argsort函数返回的是数组值从小到大的索引值
        order = []
        for i in range(self.one_dims):
            order.append(self.env.job_repeat[index[i]])  # 这个时候job编号出现的次数就等于加工的第几道工序了
        machine = []
        # machine_time = []
        operation = np.zeros(self.env.job_num, dtype=int)
        for i in order:
            job = i-1
            op = operation[job]
            choice = random.sample(range(len(self.env.machine_sign[job][op])), 1)
            # print(choice)
            machine.append(self.env.machine_sign[job][op][choice[0]])
            # machine_time.append(self.env.machine_time[job][op][choice[0]])
            operation[job] += 1
        position = order + machine
        position = np.array(position)
        # target, _, _, _ = self.env.time_count(position)
        target, _, _ = self.env.active_schedule(position)
        return [position, target]  # position是一个np.array

    # 原有函数
    # def initialize_variables(self):
    #     self.immunity_type_list = np.random.randint(0, 3, self.pop_size)  # Randint [0, 1, 2]
    #     self.age_list = np.zeros(self.pop_size)  # Control the age of each position
    #     self.finished = False

    # # 马训德改写函数
    def initialize_variables(self):
        pop_sort = deepcopy(self.pop)
        fit_list = [agent[self.FIT_ID] for agent in self.pop]
        index = np.array(fit_list).argsort()
        for i in range(len(self.pop)):
            pop_sort[i] = self.pop[index[i]]
        self.pop = deepcopy(pop_sort)
        num = int(self.pop_size / 3)
        type_list = [2, 1, 0] * num + [0] * (self.pop_size - 3 * num)
        # random.shuffle(type_list)
        self.immunity_type_list = np.array(type_list)
        self.age_list = np.zeros(self.pop_size)  # Control the age of each position

    def draw_convergence(self, repeat, name, date, current):
        d = []
        for i in range(self.epoch):
            d.append([i+1, self.list_global_best[i][self.FIT_ID]])
        d = np.array(d).reshape(len(d), 2)
        plt.plot(d[:, 0], d[:, 1])  # 画完工时间随迭代次数的变化
        font1 = {'weight': 'bold', 'size': 17}  # 汉字字体大小，可以修改
        plt.xlabel("迭代次数", font1)
        plt.title(name + "环境下的完工时间变化图", font1)
        plt.ylabel("完工时间", font1)
        folder = date + '@'+current + '_convergence_' + '@' + name
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig("./" + folder + '/' + name + "第" + str(repeat) + "次重复的convergence.jpg",
                    bbox_inches='tight', dpi=500)
        plt.savefig("./" + folder + '/' + name + "第" + str(repeat) + "次重复的convergence.svg",
                    bbox_inches='tight', dpi=500)
        # plt.ion()
        # plt.show()
        # plt.pause(2)
        plt.close()


