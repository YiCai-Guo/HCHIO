from chio_my import MY_CHIO
import scipy
from FJSP_env import Flexible
import time


# for i in [2, 4, 5, 6, 7, 10, 1, 3, 9, 8]:
for i in [8]:
# for i in [10, 20]:
    filename = '.\\DataSet\\la' + str(i).zfill(2) + '.fjs'
    # filename = '.\\DataSet\\Mk10.txt'

    name = str.split(str.split(filename, '\\')[-1], '.')[0]
    # when = time.asctime()[4:16].replace(':', '-')
    when = time.asctime()[4:16]
    date = when[0:6]
    current = when[7:12].replace(':', '-')
    with open('.\\' + date + '@' + name + '.txt', 'a') as f:
        env = Flexible(filename, name)

        """
        brr随机交流，2也随机
        order选择，p2更新，create_solution，综合性vns,0.1概率换机器，
        全算例测试mutation 0.05+0.05,
        正交实验  6
        """
        epoch = 1500
        pop_size = 200
        brr = 0.65
        max_age = 30
        f.write('\n')
        f.write(f'epoch: {epoch}, pop_size: {pop_size}, brr: {brr}, max_age: {max_age}'+'\n')
        f.write('start time: '+time.asctime()+'\n')
        s, z = 0, 0
        for repeat in range(10):
            model = MY_CHIO(env, epoch, pop_size, brr, max_age)
            # model = BaseCHIO(epoch, pop_size, brr, max_age)
            best_position, best_fitness, converge_epoch = model.solve()  # model 里继承了父类 optimizer 中的 solve 方法
            model.draw_convergence(repeat, name, date, current)
            env.draw_gantt(best_position, repeat, name, date, current)
            f.write(current + ' '+str(repeat) + " "+str(best_fitness) + ' '+str(converge_epoch)+'\n')
            s += best_fitness
            z += converge_epoch
        s = s/(repeat+1)
        z /= (repeat+1)
        f.write(current + ' '+"Average " + '%.2f' % s + ' ' + '%.2f' % z + '\n')
        f.write('finish time: '+time.asctime()+'\n'*2)

