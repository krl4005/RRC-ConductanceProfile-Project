# IMPORT FUNCTIONS
import random
from math import log10
from multiprocessing import Pool
import numpy as np
import pandas as pd
from deap import base, creator, tools
import time
from important_functions import get_last_ap, check_physio, rrc_search, get_rrc_error, get_features, check_physio_torord, run_model

# DEFINE GA_CONFIG CLASS
class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             params_lower_bound,
             params_upper_bound,
             tunable_parameters,
             mate_probability,
             mutate_probability,
             gene_swap_probability,
             gene_mutation_probability,
             tournament_size,
             cost,
             rrc_cost,
             feature_targets,
             save_data_to,
             path_to_model,
             model):
        self.population_size = population_size
        self.max_generations = max_generations
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.tunable_parameters = tunable_parameters
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.cost = cost
        self.rrc_cost = rrc_cost
        self.feature_targets = feature_targets
        self.save_data_to = save_data_to
        self.path_to_model = path_to_model
        self.model = model

# DEFINE GA FUNCTIONS
def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.
    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')    
    cols = ['gen'] + GA_CONFIG.tunable_parameters + ['fitness', 'rrc', 'rrc_error', 't', 'v', 'cai','total_feature_error', 'total_morph_error', 'feature_error_AP4', 'morph_error_AP4', 'Vm_peak_AP4', 'dvdt_max_AP4', 'apd40_AP4', 'apd50_AP4', 'apd90_AP4', 'triangulation_AP4', 'RMP_AP4', 'cat_amp_AP4', 'cat_peak_AP4', 'cat90_AP4']

    # 3. Calls _initialize_individuals and returns initial population
    population = toolbox.population(GA_CONFIG.population_size)

    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population) #this doesn't execute the code until we need it

    info = []
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit[0],)
        ind_info = [0] + list(ind[0].values()) + fit
        info.append(ind_info) 
        
    # Note: visualize individual fitnesses with: population[0].fitness
    gen_fitnesses = [ind.fitness.values[0] for ind in population]

    print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
    print(f'\tBest fitness is {np.min(gen_fitnesses)}')

    ## BELOW INCLUDES CODE TO SAVE SPECIFIC VARIABLES AS THE GA LOOPS
    final_population = [population]
    df_info = pd.DataFrame(info, columns = cols)
    df_info.to_csv(GA_CONFIG.save_data_to+'data.csv', index=False)

    for generation in range(1, GA_CONFIG.max_generations):
        old_population = population
        old_info = info

        print('Generation {}'.format(generation))
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.

        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            if random.random() < GA_CONFIG.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.
        # 8. Evaluating the offspring of the current generation
        updated_idx = [i for i in list(range(0, len(offspring))) if not offspring[i].fitness.values]
        updated_individuals = [i for i in offspring if not i.fitness.values]

        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        info = []
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit[0],)
            ind_info = [generation] + list(ind[0].values()) + fit
            info.append(ind_info) 

        population = offspring

        gen_fitnesses = [ind.fitness.values[0] for ind in population]

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')

        final_population.append(population)

        ## BELOW INCLUDES CODE TO SAVE SPECIFIC VARIABLES AS THE GA LOOPS
        for i in list(range(0, GA_CONFIG.population_size)):
            if updated_idx.count(i) == 0:
                for x in list(range(0, len(old_population))):
                    if list(old_population[x][0].values())==list(population[i][0].values()):
                        idx = x
                info.insert(i, [generation]+old_info[idx][1:len(old_info[idx])]) 
        
        #SAVE POP AND GEN AS GA LOOPS
        new_info = pd.DataFrame(info, columns = cols)
        df_info = df_info.append(new_info, ignore_index=True)
        df_info.to_csv(GA_CONFIG.save_data_to+'data.csv', index=False)
        
        print('Finished saving generation')

    return final_population

def _initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 
    Returns:
        An Individual with conductance parameters 
    """
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp) for i in range(0, len(GA_CONFIG.tunable_parameters))]

    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))

def _mate(i_one, i_two):
    """Performs crossover between two individuals.
    There may be a possibility no parameters are swapped. This probability
    is controlled by `GA_CONFIG.gene_swap_probability`. Modifies
    both individuals in-place.
    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].items():
        if random.random() < GA_CONFIG.gene_swap_probability:
            i_one[0][key],\
                i_two[0][key] = (
                    i_two[0][key],
                    i_one[0][key])

def _mutate(individual):
    """Performs a mutation on an individual in the population.
    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.
    Args:
        individual: An individual to be mutated.
    """
    keys = [k for k, v in individual[0].items()]

    for key in keys:
        if random.random() < GA_CONFIG.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_CONFIG.params_lower_bound) or
                (new_param > GA_CONFIG.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0][key],
                        individual[0][key] * .1)

            individual[0][key] = new_param

def _evaluate_fitness(ind):

    beats = 5
    all_features = []
    all_feature_errors = []
    all_morph_errors = []
    dat, IC = run_model(ind, beats, prepace = 600, path = GA_CONFIG.path_to_model, model = GA_CONFIG.model)

    for i in list(range(0, beats-1)):
        data = get_last_ap(dat, i)
        ap_features = get_features(data['t'], data['v'], data['cai'])

        if ap_features == 50000000:
            data = [50000000] * 20
            return data
        
        all_features.append(ap_features) 
    
        feature_error = check_physio(ap_features, cost = GA_CONFIG.cost, feature_targets=GA_CONFIG.feature_targets)
        all_feature_errors.append(feature_error)
        
        AP_morph_error = check_physio_torord(data['t'], data['v'], path = GA_CONFIG.path_to_model, filter = 'yes')
        all_morph_errors.append(AP_morph_error['error'])

    results = rrc_search(ind, IC, path = GA_CONFIG.path_to_model, model = GA_CONFIG.model) 
    RRC = results['RRC']

    rrc_fitness, RRC_est = get_rrc_error(RRC, GA_CONFIG.rrc_cost)
    
    fitness = sum(all_feature_errors) + sum(all_morph_errors) + rrc_fitness

    data = [fitness, RRC, rrc_fitness, str(list(data['t'])), str(list(data['v'])), str(list(data['cai'])), sum(all_feature_errors), sum(all_morph_errors), feature_error, AP_morph_error['error']] + list(ap_features.values()) 
    return data

def start_ga(pop_size=200, max_generations=100, physio_cost = 'function_1', rrc_cost = 'function_1', save_data_to = './', path_to_model = './', model = 'tor_ord_endo2.mmt', multithread = 'yes', tunable_parameters = ['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier']):
    feature_targets = {'Vm_peak': [10, 33, 55],
                       'dvdt_max': [100, 347, 1000],
                       'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                       'cat_peak': [40, 58, 60],
                       'cat90': [350, 467, 500]}

    # 1. Initializing GA hyperparameters
    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          params_lower_bound=0.33,
                          params_upper_bound=3,
                          tunable_parameters= tunable_parameters,
                          mate_probability=0.9,
                          mutate_probability=0.9,
                          gene_swap_probability=0.2,
                          gene_mutation_probability=0.2,
                          tournament_size=2,
                          cost=physio_cost,
                          rrc_cost = rrc_cost,
                          feature_targets=feature_targets,
                          save_data_to=save_data_to,
                          path_to_model=path_to_model,
                          model = model)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individuals)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    # To speed things up with multi-threading
    if multithread == 'yes':
        p = Pool()
        toolbox.register("map", p.map)
    else:
        # Use this if you don't want multi-threading
        toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population


