import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

plt.rcParams['font.sans-serif'] = ['STSong']
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class Flexible:
    def __init__(self, file_path, name):
        self.job_num, self.machine_num, self.machine_sign, self.machine_time, self.job_repeat = self.data_read(file_path)
        self.file_path = file_path
        self.dataset_name = name
        # print(f'job_repeat: {len(self.job_repeat)}')
        # for i in range(self.job_num):
        #     print(f"第{i + 1}个job：**********-----------------*********")
        #     print('全部工序可选机器集：', self.machine_sign[i])
        #     print('可选机器的加工时间：', self.machine_time[i])

    @staticmethod
    def data_read(file_path):
        """
        先读取文件名才能获得，lb与ub的信息，编码长度等

        Returns:
            self.job_num: int类型，数据集中包含job的总数
            self.machine_num: int类型，数据集中包含machine的总数
            self.machine_sign: list类型，三维list，
                              [ job1的工序可选机器:[[job1的op1的可选机器集合]，[job1的op2的可选机器集合]，……]
                                job2的工序可选机器:[[job2的op1的可选机器集合]，[job2的op2的可选机器集合]，……] ]
            self.machine_time: list类型，三维list，
                              [ job1的机器对应时间:[[job1的op1的机器对应时间]，[job1的op2的机器对应时间]，……]
                                job2的机器对应时间:[[job2的op1的机器对应时间]，[job2的op2的机器对应时间]，……] ]
            self.job_repeat:
        """
        f = open(file_path)
        fr = f.readlines()
        int_list = []
        for line in fr:
            line2 = line.strip()
            temp = []
            s = ''
            for i in range(len(line2)):
                if line2[i].isdigit():
                    s += line2[i]
                    if i == len(line2) - 1:
                        temp.append(int(s))
                        s = ''
                else:
                    if len(s) != 0:
                        temp.append(int(s))
                        s = ''
            if len(temp) != 0:
                int_list.append(temp)
        f.close()

        job_num = int_list[0][0]
        machine_num = int_list[0][1]
        int_list = int_list[1:len(int_list)]
        machine_sign = []  # 二维列表，[[[job1工序1的可选机器集],[job1工序2的可选机器集],[],……],[工件2],……]元素为机器的编号(从1开始)
        machine_time = []  # 二维列表，与上列表所对应的机器加工时间
        for i in range(job_num):
            machine_sign.append([])
            machine_time.append([])
        job_repeat = []  # 一维列表，job加工顺序，job编号(从1开始计数)重复出现次数即为对应job的第几道工序
        for i in range(job_num):
            op_num = int_list[i][0]
            temp = int_list[i][1:]
            # print(temp)
            job_repeat += [i + 1] * op_num
            # print(job_order)
            for j in range(op_num):
                index = 0
                choices_num = temp[index]
                next_index = index + 1 + 2 * choices_num
                machine_temp = []
                time_temp = []
                for k in range(choices_num):
                    machine_temp.append(temp[1+k*2])
                    time_temp.append(temp[2+2*k])
                del temp[index:next_index]
                machine_sign[i].append(machine_temp)
                machine_time[i].append(time_temp)
        return job_num, machine_num, machine_sign, machine_time, job_repeat

    def fitness_function(self, solution):

        order, machine_res, machine_res_time = self.discrete(solution)
        make_span, _, _, _, _, _ = self.time_count(order, machine_res, machine_res_time)
        return make_span

    def time_count(self, order):
        """
        计数最大完工时间。
        """
        job_next_time = np.zeros(self.job_num)
        machine_next_time = np.zeros(self.machine_num)
        job_operation = np.zeros(self.job_num, dtype=int)
        list_start_time = []
        work_time = []
        for i in range(len(self.job_repeat)):
            job = order[i] - 1  # 加工的job编号，转化为从0开始
            op = job_operation[job]  # 该job加工,工序的位置索引
            job_operation[job] += 1
            machine_num = order[len(self.job_repeat) + i]  # 加工的machine编号
            machine = machine_num - 1  # machine编号转化为从0开始
            pos = self.machine_sign[job][op].index(machine_num)
            time = self.machine_time[job][op][pos]
            work_time.append(time)
            start_time = max(job_next_time[job], machine_next_time[machine])
            machine_next_time[machine] = start_time + time
            job_next_time[job] = start_time + time

            list_start_time.append(start_time)
        last_machine_mark = np.argmax(machine_next_time) + 1  # 结束最晚的机器
        c_finish = max(machine_next_time)  # 最晚完工时间
        return c_finish, list_start_time, work_time, last_machine_mark

    def draw(self, best_position, epoch, name, when):  # 画图
        c_finish, list_start_time, work_time, last_machine_mark = self.time_count(best_position)
        # print(work_time)
        # print(type(work_time))
        # figure, ax = plt.subplots(figsize=(48, 8))
        figure, ax = plt.subplots(figsize=(16, 8))
        count = np.ones(self.job_num)  # 记录job已经被加工到第几个工序了
        job_color = np.random.rand(self.job_num, 3)
        for i in range(len(self.job_repeat)):  # 每一道工序画一个小框
            plt.bar(x=list_start_time[i], bottom=best_position[i+len(self.job_repeat)], height=0.5, width=work_time[i],
                    orientation="horizontal",
                    color=job_color[best_position[i] - 1], edgecolor='black')
            plt.text(list_start_time[i] + work_time[i] / 2, best_position[i+len(self.job_repeat)] - 0.5,
                     '%.0f' % (work_time[i]), color='blue', fontsize=10, weight='bold', va='center', ha='center')

            plt.text(list_start_time[i] + work_time[i] / 2, best_position[i+len(self.job_repeat)],
                     '%.0f-%.0f' % (best_position[i], count[best_position[i] - 1]), color='black', fontsize=9,
                     weight='bold', va='center', ha='center')  # 12是矩形框里字体的大小，可修改
            # plt.text(list_start_time[i] + machine_res_time[i] / 2, machine_res[i],
            #          '%.0f' % (order[i]), color='black', fontsize=10,
            #          weight='bold', va='center', ha='center')  # 12是矩形框里字体的大小，可修改

            count[best_position[i] - 1] += 1

        plt.plot([c_finish, c_finish], [0, last_machine_mark], c='black', linestyle='-.',
                 label='完工时间=%.1f' % c_finish)  # 用虚线画出最晚完工时间，并添加图例label
        plt.plot(c='blue', label='完工时间=%.1f' % c_finish)
        plt.legend(prop={'family': ['STSong'], 'size': 16})  # 图例字体大小，可以修改

        font1 = {'weight': 'bold', 'size': 17}  # 汉字字体大小，可以修改
        plt.xlabel("加工时间", font1)
        plt.title(f"{name}环境下"+"第"+str(epoch)+"回合的甘特图", font1)
        plt.ylabel("机器", font1)

        position, mark = [], []  # y轴的标签和位置
        for i in range(self.machine_num):
            position.append(i + 1)
            s = "M" + str(i + 1)
            mark.append(s)
        plt.yticks(position, mark)

        plt.axis([0, c_finish * 1.01, 0, self.machine_num + 1])  # 设定每个轴的大小范围
        plt.tick_params(labelsize=17)  # 坐标轴刻度字体大小，可以修改
        labels = ax.get_xticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # plt.ion()
        folder = 'gantt_' + name + '@' + when
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig("./"+folder+'/'+name+"第"+str(epoch)+"回合的gantt.jpg", bbox_inches='tight', dpi=500)
        # plt.show()
        # plt.pause(1)
        plt.close()

    def active_schedule(self, order):
        """
        主动调度解码：安排工序到某机器时，先遍历此机器的加工间隔，对比此工件的上一道工序完成时间，能插入则插入
        Args：
            order：一维，np.array或list类型，对应CHIO算法应为np.array类型，两段式编码，长度为：2*len(self.job_repeat)
        Returns:
            makespan：float类型，最大完工时间
            last_machine_list：int类型，最后时刻 makespan才完成任务的机器编号构成的列表，从1开始计数机器编号
            machine_record：三维list，job(工件)从1开始，op(该工件的工序)从1开始，
                            [job, op, (起始时间，终止时间）]元素均按时间顺序正序排列，终止时间0>起始时间1
                            [ 机器1的任务：[[job0, op0, (起始时间0，终止时间0)]，[job1, op1, (起始时间1，终止时间1)], ...],
                              机器2的任务：[[job2, op2, (起始时间2，终止时间2)]，[job3, op3, (起始时间3，终止时间3)], ...], ...]
        """
        job_operation = np.zeros(self.job_num, dtype=int)       # 记录是该工件的第几道工序以访问：机器及加工时间等信息
        job_release_time = np.zeros(self.job_num)               # 工件的上一道工序何时完成，即释放时间
        machine_record = [[] for i in range(self.machine_num)]
        # # machine_record:记录机器间隔：[[job,op,(start,end)],……]
        for i in range(len(self.job_repeat)):
            job = order[i]-1          # order 前半段工序编码部分，job号从1开始计数
            op = job_operation[job]  # 该job即将加工的工序的可选机器集合在machine_sign中的位置索引，从0计数，对应工序1
            machine_num = order[len(self.job_repeat)+i]           # order的后半段编码
            # print(f'i:{i},job:{job},op:{op}')
            # print(self.machine_sign[job][op])
            # print(machine_num)
            # print(order)
            pos = self.machine_sign[job][op].index(machine_num)
            time = self.machine_time[job][op][pos]
            # print(type(time))
            machine = machine_num-1                               # machine编码部分也是从1开始
            if len(machine_record[machine]) == 0:   # 机器上未安排任何工序
                machine_record[machine].append([job + 1, op+1, (job_release_time[job], job_release_time[job] + time)])
                job_release_time[job] = job_release_time[job] + time
                job_operation[job] += 1
            elif len(machine_record[machine]) == 1:       # 机器上只有一个工序任务
                e = machine_record[machine][0][2][0]
                s = max(0, job_release_time[job])
                if e-s >= time:     # 可以插到前面
                    machine_record[machine].insert(0, [job + 1, op + 1, (s, s + time)])
                    job_release_time[job] = s + time
                    job_operation[job] += 1
                    continue
                else:     # 间隔不够，只能插到后面
                    s = max(job_release_time[job], machine_record[machine][0][2][1])
                    machine_record[machine].append([job + 1, op + 1, (s, s + time)])
                    job_release_time[job] = s + time
                    job_operation[job] += 1
            else:
                #  头插检验
                en = machine_record[machine][0][2][0]
                st = max(0, job_release_time[job])
                if en - st >= time:
                    machine_record[machine].insert(0, [job + 1, op + 1, (st, st + time)])
                    job_release_time[job] = st + time
                    job_operation[job] += 1
                    continue
                #  两任务之间的间隔检查
                start = 0
                flag = False
                index = 0
                for j in range(len(machine_record[machine])-1):
                    s = machine_record[machine][j][2][1]
                    s = max(s, job_release_time[job])
                    e = machine_record[machine][j+1][2][0]
                    if e-s >= time:
                        flag = True
                        index = j+1
                        start = s
                        break
                if flag:  # 找到合适间隔
                    machine_record[machine].insert(index, [job+1, op+1, (start, start+time)])
                else:     # 未找到合适间隔，只能插入尾部
                    start = max(machine_record[machine][-1][2][1], job_release_time[job])
                    machine_record[machine].append([job+1, op+1, (start, start+time)])
                job_release_time[job] = start + time
                job_operation[job] += 1

        last_machine_list = []
        finish_time = []
        for i in range(self.machine_num):    # 记录每一个机器的完工时间
            if len(machine_record[i]) != 0:
                finish_time.append(machine_record[i][-1][2][1])
            else:  # 有的机器可能最后一道工序任务都没有，那么machine_record[i][-1]就访问错误了
                continue
        makespan = max(finish_time)
        for i in range(self.machine_num):    # 收集最后才完成的机器编号
            if len(machine_record[i]) != 0 and machine_record[i][-1][2][1] == makespan:
                last_machine_list.append(i+1)  # 机器编号从1开始计数
            else:  # 有的机器可能最后一道工序任务都没有，那么machine_record[i][-1]就访问错误了
                continue
        return makespan, last_machine_list, machine_record

    def draw_gantt(self, best_position, repeat, name, date, current):
        """
        本函数内所有变量均对应于：active_schedule所产生的machine_record的画图操作
        Args：
            best——position： 一维，np.array或list类型，对应CHIO算法应为np.array类型，两段式编码，长度为：2*len(self.job_repeat)
        Returns：
            保存图片到 gantt_Mk+数据集名字+日期+时-分 的文件夹中
        """
        makespan, last_machine_list, machine_record = self.active_schedule(best_position)
        # figure, ax = plt.subplots(figsize=(48, 8))
        figure, ax = plt.subplots(figsize=(16, 8))
        job_color = np.random.rand(self.job_num, 3)
        for i in range(self.machine_num):  # 每一道工序画一个小框
            for op_info in machine_record[i]:
                plt.bar(x=op_info[2][0], bottom=i+1, height=0.5, width=op_info[2][1]-op_info[2][0],
                        orientation="horizontal", color=job_color[op_info[0] - 1], edgecolor='black')
                plt.text((op_info[2][0]+op_info[2][1]) / 2, i+0.5, '%.0f' % (op_info[2][1]-op_info[2][0]),
                         color='blue', fontsize=10, weight='bold', va='center', ha='center')
                # plt.text((op_info[2][0]+op_info[2][1]) / 2, i+1,
                #          '%.0f-%.0f' % (op_info[0], op_info[1]), color='black', fontsize=9,
                #          weight='bold', va='center', ha='center')  # 9是矩形框里字体的大小，可修改
                plt.text((op_info[2][0] + op_info[2][1]) / 2, i + 1,
                         '%.0f' % (op_info[0]), color='black', fontsize=9,
                         weight='bold', va='center', ha='center')  # 9是矩形框里字体的大小，可修改
        start = 0
        plt.plot([makespan, makespan], [start, last_machine_list[0]], c='black', linestyle='-.',
                 label='完工时间=%.1f' % makespan)  # 用虚线画出最晚完工时间，并添加图例label
        start = last_machine_list[0]
        for machine_No in last_machine_list:  # 第一次循环体内，只是一个点，不会画线的
            plt.plot([makespan, makespan], [start, machine_No], c='black', linestyle='-.')  # 用虚线画出最晚完工时间，不添加图例label
        plt.legend(prop={'family': ['STSong'], 'size': 16})  # 图例字体大小，可以修改

        font1 = {'weight': 'bold', 'size': 17}  # 汉字字体大小，可以修改
        plt.xlabel("加工时间", font1)
        plt.title(f"{name}环境下" + "的甘特图", font1)
        # plt.title(f"{name}环境下" + "第" + str(repeat) + "回合的甘特图", font1)
        plt.ylabel("机器", font1)

        position, mark = [], []  # y轴的标签和位置
        for i in range(self.machine_num):
            position.append(i + 1)
            s = "M" + str(i + 1)
            mark.append(s)
        plt.yticks(position, mark)

        plt.axis([0, makespan * 1.01, 0, self.machine_num + 1])  # 设定每个轴的大小范围
        plt.tick_params(labelsize=17)  # 坐标轴刻度字体大小，可以修改
        labels = ax.get_xticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        folder = date+'@'+current + '_gantt_' + '@' + name
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig("./" + folder + '/' + name + "第" + str(repeat) + "次重复的gantt.jpg", bbox_inches='tight', dpi=500)

        plt.savefig("./" + folder + '/' + name + "第" + str(repeat) + "次重复的gantt.svg", bbox_inches='tight', dpi=500,  format="svg")

        # plt.ion()
        # plt.show()
        # plt.pause(2)
        plt.close()






