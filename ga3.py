import copy
import random
import numpy as np
import matplotlib.pyplot as plt

data_path = "./data/had12.dat"
data_file = open(data_path, 'r')
n_facilities = int(data_file.readline().lstrip().rstrip())
data_file.close()

data_matrix = np.loadtxt(data_path, skiprows=2)
distance_matrix = data_matrix[:n_facilities]
flow_matrix = data_matrix[n_facilities:]

population_size = 1000
crossover_probability = 0.8
mutation_probability = 0.008


def generate_random_population(n_facilities, n_chromosomes):
	population_list = []
	for ch_idx in range(n_chromosomes):
		rand_chromosome = list(range(0, n_facilities))
		random.shuffle(rand_chromosome)
		population_list.append(rand_chromosome)
	return population_list


def fitness_score(population_list, flow_matrix, distance_matrix):
	n_facilities = len(population_list[0])
	fitness_score_list = []
	for chromosome_list in population_list:
		chromosome_fitness = 0
		for loc_idx_f in range(0, n_facilities):
			fac_und_loc_f = chromosome_list[loc_idx_f]
			for loc_idx_s in range(0, n_facilities):
				fac_und_loc_s = chromosome_list[loc_idx_s]
				ft_s = flow_matrix[fac_und_loc_f, fac_und_loc_s] * distance_matrix[loc_idx_f, loc_idx_s]
				chromosome_fitness += ft_s
		fitness_score_list.append(1. / (chromosome_fitness / 2.))
	return fitness_score_list


print(1. / fitness_score([[4, 3, 6, 0, 1, 2, 5]], flow_matrix, distance_matrix)[0])


def normalise_fitness_score(fitness_score):
	return np.array(fitness_score) / np.sum(fitness_score)


def roulette_selection(population_list, fitness_scores_list, elitism=True):
	new_species = []
	population_size = len(fitness_scores_list)
	population_size = population_size - 1 if elitism else population_size
	cum_sum = np.cumsum(fitness_scores_list, axis=0)
	for _ in range(0, population_size):
		rnd = random.uniform(0, 1)
		if rnd < cum_sum[0]:
			new_species.append(population_list[0])
			continue
		counter = 0
		while rnd > cum_sum[counter]:
			counter += 1
		new_species.append(population_list[counter])
	new_species.append(population_list[np.argmax(fitness_scores_list)])
	return new_species


def chromosome_crossover(chromosome_o, chromosome_s):
	chr_o, chr_s = copy.copy(chromosome_o), copy.copy(chromosome_s)
	pos = random.randint(0, len(chromosome_o) - 1)
	for ch_idx in range(0, pos):
		fac_o = chr_o[ch_idx]
		fac_s = chr_s[ch_idx]
		fac_os_idx = chr_o.index(fac_s)
		fac_so_idx = chr_s.index(fac_o)
		chr_o[fac_os_idx] = fac_o
		chr_s[fac_so_idx] = fac_s
		chr_o[ch_idx] = fac_s
		chr_s[ch_idx] = fac_o
	return chr_o, chr_s


def crossover_population(new_species, crossover_probability):
	species_nc = []
	crossover_list = []
	for n_chrom in new_species:
		rnd = random.uniform(0, 1)
		if rnd < crossover_list:
			crossover_list.append(n_chrom)
		else:
			species_nc.append(n_chrom)
	crossover_tuples = []
	cr_iterate = list(enumerate(crossover_list))
	while cr_iterate:
		cch_idx, c_chrom = cr_iterate.pop()
		if not cr_iterate:
			species_nc.append(c_chrom)
			break
		cb_idx, cross_buddy = random.choice(cr_iterate)
		cr_iterate = [(x_k, x_v) for x_k, x_v in cr_iterate if x_k != cb_idx]
		crossover_tuples.append((c_chrom, cross_buddy))
	after_cover = []
	for cr_tup in crossover_tuples:
		cr_o, cr_t = chromosome_crossover(cr_tup[0], cr_tup[1])
		after_cover.append(cr_o)
		after_cover.append(cr_t)
	new_species = after_cover + species_nc
	return new_species


def mutation_population(new_species, mutation_probability):
	mutated = []
	for chromosome in new_species:
		for b_idx in range(0, len(chromosome)):
			rnd = random.uniform(0, 1)
			if rnd < mutation_probability:
				swap_idx = random.randint(0, len(chromosome) - 1)
				old_mut_val = chromosome[b_idx]
				chromosome[b_idx] = chromosome[swap_idx]
				chromosome[swap_idx] = old_mut_val
		mutated.append(chromosome)
	return mutated


poplation_list = generate_random_population(n_facilities, population_size)
for epoch in range(0, 100000):
	fit_scores = fitness_score(poplation_list, flow_matrix, distance_matrix)
	fit_scores_norm = normalise_fitness_score(fit_scores)
	selected_ch = roulette_selection(poplation_list, fit_scores_norm, elitism=True)
	crossed_ch = crossover_population(selected_ch, crossover_probability)
	mutated_ch = mutation_probability(crossed_ch, mutation_probability)
	max_fitness = np.max(fit_scores)
	max_chromosome = poplation_list[np.argmax(fit_scores)]
	max_chromosome = [x + 1 for x in max_chromosome]
	print("Epoch: {}, Population fitness score: {}, Max score: {}, Max chromosome: {}".format(epoch,
	                                                                                          1. / np.mean(fit_scores),
	                                                                                          1. / max_fitness,
	                                                                                          max_chromosome))
	population_list = crossed_ch
