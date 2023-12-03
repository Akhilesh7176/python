
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import logging

log = logging.getLogger()


start_value = -0.1
stop_value = [1.1, 60.1]
step_value = [0.05, 5]
trapmf_y_value_smf = [[-2.0, -1.5, 0.05, 0.8],[0, 30, 70, 80] , [-2, -1, 0.1, 0.2], [0.3, 0.5, 2, 4],
                      [-0.5, -0.3, 0.05, 0.2], [-0.5, -0.3, 0.1,
                                                0.2], [0.1, 0.9, 2, 4], [15, 30, 70, 80],
                      [-0.5, -0.3, 0.05, 0.9]]

trapmf_y_value_mmf = [[0, 0.6, 0.7, 1], [0, 10, 30], [0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.5],
                      [0, 0.2, 0.3, 0.5], [0.1, 0.2, 0.4, 0.6], [
                          0.05, 0.8, 2, 4], [5, 15, 30],
                      [0, 0.2, 1]]

trapmf_y_value_lmf = [[0.3, 0.8, 1.5, 2], [-20, -10, 0, 15], [0.3, 0.5, 2, 4], [-2, -1, 0.1, 0.2],
                      [0, 0.5, 2, 4], [0.4, 0.6, 2, 4], [-0.5, -
                                                         0.3, 0.05, 0.8], [-10, -5, 5, 15],
                      [0, 0.6, 2, 4]]


group_smf_mmf_lmf = []


def func_from_deap_GA(value_from_GA):
    log.info('Entering into membership func - ', value_from_GA)
    # Entering into the membership function
    for value in range(len(value_from_GA)):
        if (value == 0):
            for item in range(9):
                if (item == 0 or item == 6):
                    if (item == 1 or item == 7):
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[0], start_value, stop_value[1], step_value[1], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))
                    else:
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[0], start_value, stop_value[0], step_value[0], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))

        elif (value == 1):
            for item in range(9):
                if (item == 2 or item == 3 or item == 8):
                    if (item == 1 or item == 7):
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[1], start_value, stop_value[1], step_value[1], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))
                    else:
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[1], start_value, stop_value[0], step_value[0], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))

        elif (value == 2):
            for item in range(9):
                if (item == 1 or item == 7):
                    if (item == 1 or item == 7):
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[2], start_value, stop_value[1], step_value[1], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))
                    else:
                        group_smf_mmf_lmf.append(membership_function(
                            item, value_from_GA[2], start_value, stop_value[0], step_value[0], trapmf_y_value_smf[item], trapmf_y_value_mmf[item], trapmf_y_value_lmf[item]))

    log.info('Group the smf_lmf_mmf', group_smf_mmf_lmf)
    # group_smf_mmf_lmf
    Pc_value = create_Pc(group_smf_mmf_lmf)
    Pm_value = create_Pm(group_smf_mmf_lmf)
    # Lrange_value = create_L_Range(group_smf_mmf_lmf)

    group_smf_mmf_lmf.clear()
    return Pc_value, Pm_value



def membership_function(item, local_value_from_GA, start, stop, step, y_smf, y_mmf, y_lmf):

    x = local_value_from_GA*np.ones(3)

    smf = fuzz.trapmf(x, y_smf)

    if (item == 1 or item == 7 or item == 8):
        mmf = fuzz.trimf(x, y_mmf)

    else:
        mmf = fuzz.trapmf(x, y_mmf)

    lmf = fuzz.trapmf(x, y_lmf)

    # plot_the_graph(x, smf, mmf, lmf)

    return round(smf[0], 4), round(mmf[0], 4), round(lmf[0], 4)


def create_Pc(smf_mmf_lmf):
    Pc_array = []
    average_Pc = []

    Pc_array.append(smf_mmf_lmf[0])
    Pc_array.append(smf_mmf_lmf[5])
    Pc_array.append(smf_mmf_lmf[2])

    num_arrays = len(Pc_array)

    num_elements = len(Pc_array[0])

    averages = [0] * num_elements

    for i in range(num_elements):
        total = 0
        for j in range(num_arrays):
            total += Pc_array[j][i]
        averages[i] = total / num_arrays

    average_Pc = [round(avg, 4) for avg in averages]

    Pc_value = ((0.4*(average_Pc[0])+(0.7*average_Pc[1])+average_Pc[2]))
    log.info('Value of the Crossover rate Array', Pc_array)
    log.info('Value of the Crossover rate Average', average_Pc)
    log.info('Value of the Crossover rate', Pc_value)

    return Pc_value


def create_Pm(smf_mmf_lmf):
    Pm_array = []
    average_Pm = []

    Pm_array.append(smf_mmf_lmf[0])
    Pm_array.append(smf_mmf_lmf[5])
    Pm_array.append(smf_mmf_lmf[3])

    num_arrays = len(Pm_array)

    num_elements = len(Pm_array[0])

    averages = [0] * num_elements

    for i in range(num_elements):
        total = 0
        for j in range(num_arrays):
            total += Pm_array[j][i]
        averages[i] = total / num_arrays

    average_Pm = [round(avg, 4) for avg in averages]

    Pm_value = ((0.1*(average_Pm[0])+(0.4*average_Pm[1])+(0.8*average_Pm[2])))
    log.info('Value of the Mutation rate Array', Pm_array)
    log.info('Value of the Mutation rate Average', average_Pm)
    log.info('Value of the Mutation rate', Pm_value)

    return Pm_value


def create_L_Range(smf_mmf_lmf):
    Lrange_array = []
    average_Lrange = []

    Lrange_array.append(smf_mmf_lmf[1])
    Lrange_array.append(smf_mmf_lmf[6])
    Lrange_array.append(smf_mmf_lmf[4])

    num_arrays = len(Lrange_array)

    num_elements = len(Lrange_array[0])

    averages = [0] * num_elements

    for i in range(num_elements):
        total = 0
        for j in range(num_arrays):
            total += Lrange_array[j][i]
        averages[i] = total / num_arrays

    average_Lrange = [round(avg, 4) for avg in averages]

    Lrange_value = 0

    minimal_lrange = min(average_Lrange)
    if (average_Lrange.index(minimal_lrange) == 0):
        Lrange_value = 0.4 * 0.9 * minimal_lrange
    elif (average_Lrange.index(minimal_lrange) == 1):
        Lrange_value = 0.8 * 0.9 *  minimal_lrange
    else:
        Lrange_value = 0.9 * minimal_lrange

   

    log.info('Lrange_array', Lrange_array)
    log.info('Average  average_Lrange', average_Lrange)
    log.info('Lrange value', Lrange_value)

    return Lrange_value


def plot_the_graph(x, smf, mmf, lmf):

    fig_scale = 1.5
    plt.figure(figsize=(6.4 * fig_scale, 4.8 * fig_scale))
    plt.figure(1)
    plt.subplot(311)
    plt.title("Fuzzy Membership Function Ncg/Ng to Pc and Pm")
    plt.plot(x, smf, label="S")
    plt.plot(x, mmf, label="M")
    plt.plot(x, lmf, label="L")

    plt.legend(loc="upper right")
    # plt.show()

    return

