import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
import pickle
import asyncio
import matplotlib as mpl
from   matplotlib.colors import ListedColormap as lcm
from scipy.ndimage import zoom
from PIL import Image

#async wrapper function, it's a decorator for running in parallel
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def plot_snapshot(matrix, identifier_string, temperature, step, length):
    figure = plt.figure()
    axes = figure.add_subplot(111)

    figure.suptitle(r'$m = {:.2f}$'.format(calculate_magnetization(matrix, length)))
    # using the matshow() function
    cax = axes.matshow(matrix, cmap='hot', interpolation='nearest', rasterized=True)

    plt.axis('off')

    path_folder = 'Figures/temperature={:.6f}/snapshots/'.format(temperature) + identifier_string
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    mpl.rcParams['text.color'] = 'white'

    figure.patch.set_facecolor('xkcd:black')

    plt.savefig(path_folder + '/step=' + str(step) + '-snapshot.png', dpi=400, bbox_inches = 'tight')

    plt.clf()
    plt.close()

def plot_magnetization_per_step(magnetizations, identifier_string):
    plt.plot(magnetizations)

    path_folder = 'Figures/temperature={:.6f}/magnetization/'.format(temperature)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    with open(path_folder + 'magnetizations_per_step' + identifier_string, "wb") as fp:   #Pickling
        pickle.dump(magnetizations, fp)

    plt.savefig(path_folder + 'magnetization_per_step' + identifier_string + '.png')
    plt.clf()
    plt.close()

def plot_magnetization_versus_temperature(range_of_temps, list_of_magnetizations, identifier_string):
    plt.scatter(range_of_temps, list_of_magnetizations)

    path_folder = 'Figures/magnetization/'
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    with open(path_folder + 'magnetization_versus_temperature' + identifier_string, "wb") as fp:   #Pickling
        pickle.dump([range_of_temps, list_of_magnetizations], fp)

    plt.savefig(path_folder + 'magnetization_versus_temperature' + identifier_string + '.png', dpi=400)
    plt.clf()
    plt.close()

def calculate_magnetization(matrix, length):
    #vector_form = np.vectorize(matrix)

    sum_cos = np.sum((np.cos(matrix * (2 * math.pi / 255))))
    sum_sin = np.sum((np.sin(matrix * (2 * math.pi / 255))))

    return math.sqrt(sum_cos**2 + sum_sin**2)/(length*length)

def coupling_constant(spin_angle, angle_between_spin_neighbour, theta):
    see_spin_variable = min(2*math.pi - abs(spin_angle - angle_between_spin_neighbour),
        abs(spin_angle - angle_between_spin_neighbour))

    if see_spin_variable <= theta / 2:
        return 1
    else:
        return 0

def metropolis(delta_energy, temperature):
    return np.exp(-delta_energy / temperature)

def glauber(delta_energy, T):
    return 0.5*(1-math.tanh((delta_energy)/(2*T)))

#@background
def simulation_for_temperature(matrix, temperature, total_mc_steps, algorithm: str):
    length         = len(matrix[0])
    # igual a 330º, usado no artigo
    theta          = 2*math.pi
    total_samples_in_step = (length * length)
    matrix_of_spins         = matrix
    magnetizations_per_step = []

    #sempre verdadeiro pra rede quadrada
    angle_between_spin_and_neighbour_below = (3*math.pi)/2
    angle_between_spin_and_neighbour_left  = math.pi #(2*math.pi)/2
    angle_between_spin_and_neighbour_above = (math.pi)/2
    angle_between_spin_and_neighbour_right = 0 #(0*math.pi)/2

    identifier_string = 'temperature=' + str(temperature) + '-algorithm=' + algorithm \
                        + '-total_mc_steps=' + str(total_mc_steps)

    for step in range(total_mc_steps):
        #if ((step % 25 == 0)) or (step < 100):
        #    plot_snapshot(matrix_of_spins, identifier_string, temperature, step, length)
        for sample in range(total_samples_in_step):
            sampled_row    = np.random.randint(0, length)
            sampled_column = np.random.randint(0, length)

            sampled_spin = matrix_of_spins[sampled_row][sampled_column] * (2 * math.pi / 255)

            #vizinho abaixo
            spin_neighbour_below = matrix_of_spins[(sampled_row + 1) % length][sampled_column]
            #vizinho acima
            spin_neighbour_above = matrix_of_spins[sampled_row - 1][sampled_column]
            #vizinho a direita
            spin_neighbour_right = matrix_of_spins[sampled_row][(sampled_column + 1) % length]
            #vizinho a esquerda
            spin_neighbour_left  = matrix_of_spins[sampled_row][sampled_column - 1]

            total_energy_sampled = - ( math.cos(sampled_spin - spin_neighbour_below) \
                                 + math.cos(sampled_spin - spin_neighbour_right) \
                                 + math.cos(sampled_spin - spin_neighbour_left) \
                                 + math.cos(sampled_spin - spin_neighbour_above))

            new_spin_candidate = np.random.uniform(0, 2 * math.pi)

            total_energy_new    = - ( math.cos(new_spin_candidate - spin_neighbour_below) \
                                + math.cos(new_spin_candidate - spin_neighbour_right) \
                                + math.cos(new_spin_candidate - spin_neighbour_left) \
                                + math.cos(new_spin_candidate - spin_neighbour_above))

            #glauber transition rate
            #if np.random.uniform(0, 1) > glauber(total_energy_sampled, total_energy_new, temperature):
            delta_energy = total_energy_new - total_energy_sampled
            if algorithm == 'glauber':
                if np.random.uniform(0, 1) < glauber(delta_energy, temperature):
                    matrix_of_spins[sampled_row][sampled_column] = new_spin_candidate
            elif algorithm == 'metropolis':
                if delta_energy <= 0:
                    #print(total_energy_new, total_energy_sampled, glauber(total_energy_sampled, total_energy_new, temperature))

                    matrix_of_spins[sampled_row][sampled_column] = int(new_spin_candidate * (255 / (2 * math.pi)))

                else:
                    if np.random.uniform(0, 1) < metropolis(delta_energy, temperature):

                        matrix_of_spins[sampled_row][sampled_column] = int(new_spin_candidate * (255 / (2 * math.pi)))
                        magnetizations_per_step.append(calculate_magnetization(matrix_of_spins, length))

    plot_magnetization_per_step(magnetizations_per_step, identifier_string)
    print('Simulation for temperature T={:.6f} finished.'.format(temperature))
    return np.mean(magnetizations_per_step[-100])

list_of_magnetizations = []
range_of_temps         = np.arange(0.2, 1.4, 0.1)
algorithm              = 'metropolis'
temperature            = 0.10
total_steps            = 5000

start_time = time.time()

srcImage = Image.open("monalisa-higher-res.png")
grayImage = srcImage.convert('L')
array = np.array(grayImage)

magnetization = simulation_for_temperature(array, temperature, algorithm, total_steps)


#plot_magnetization_versus_temperature(range_of_temps, list_of_magnetizations, 'algorithm=' + algorithm
#    + 'range_of_temperatures:' + str(range_of_temps[0]) + 'to' + str(range_of_temps[-1]))
