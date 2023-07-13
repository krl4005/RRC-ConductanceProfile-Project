# IMPORT FUNCTIONS
from important_functions import run_model, get_ind, rrc_search, check_robustness, collect_rrc_data
from multiprocessing import Pool
import time
import pandas as pd
import pickle

def get_baseline_torord_data(save_to = './data/baseline_torord_data.csv'):
    dat, IC = run_model(None, 1)
    baseline_torord = pd.DataFrame({'t': dat['engine.time'], 'v': dat['membrane.v']})
    baseline_torord.to_csv(save_to, index = False)

def get_cond_data(best_data_path = './data/best_data.csv', save_to = './data/cond_data.pkl'):
    """
    This function runs simulations in the ToR-ORd and Grandi models to generate action potential data. 
    Specically, this generates the data in the following manuscript figures: 4A, 5, 6B, 7, 8A, and 8B. 
    The data is stored as a pickle file contained in the data/ folder and named cond_data.pkl.
    """
    
    # LOAD DATA
    best_data = pd.read_csv(best_data_path)

    ##########################################################################################################################################################
    # RUN SIMULATIONS

    # Baseline Data - BM
    base_ind = get_ind()
    base_ind['i_bias_multiplier'] = 0
    base_ind['i_bias1_multiplier'] = 0
    dat, IC = run_model([base_ind], 1)
    result = rrc_search([base_ind], IC)
    all_data = {**base_ind, **result, **{'dat':dat}}

    # Optimized Data - OM
    ind = best_data.iloc[0].filter(like = 'multiplier').to_dict()
    ind['i_bias_multiplier'] = 0
    ind['i_bias1_multiplier'] = 0
    dat_o, IC_o = run_model([ind], 1)
    result_o = rrc_search([ind], IC_o)
    all_data_o = {**ind, **result_o, **{'dat':dat_o}}

    # Optimized Data without INaL - OM1
    ind_0 = ind.copy()
    ind_0['i_nal_multiplier'] = 0
    dat_0, IC_0 = run_model([ind_0], 1)
    result_0 = rrc_search([ind_0], IC_0)
    all_data_0 = {**ind_0, **result_0, **{'dat':dat_0}}

    # Optimized Data without INaL and ICaL Rescue - OM2
    ind_1 = ind_0.copy()
    ind_1['i_cal_pca_multiplier'] = 3
    dat_1, IC_1 = run_model([ind_1], 1)
    result_1 = rrc_search([ind_1], IC_1)
    all_data_1 = {**ind_1, **result_1, **{'dat':dat_1}}

    # Optimized Data without ICaL - OM3
    ind_0a = ind.copy()
    ind_0a['i_cal_pca_multiplier'] = 0
    dat_0a, IC_0a = run_model([ind_0a], 1)
    result_0a = rrc_search([ind_0a], IC_0a)
    all_data_0a = {**ind_0a, **result_0a, **{'dat':dat_0a}}

    # Baseline with fake outward - BM2
    ind_2 = base_ind.copy()
    ind_2['i_bias_multiplier'] = 0.8
    ind_2['i_bias1_multiplier'] = 0
    dat_2, IC_2 = run_model([ind_2], 1)
    result_2 = rrc_search([ind_2], IC_2)
    all_data_2 = {**ind_2, **result_2, **{'dat':dat_2}}

    # Baseline with fake outward and ICaL = 3 - BM3
    ind_3 = ind_2.copy()
    ind_3['i_cal_pca_multiplier'] = 3
    dat_3, IC_3 = run_model([ind_3], 1)
    result_3 = rrc_search([ind_3], IC_3)
    all_data_3 = {**ind_3, **result_3, **{'dat':dat_3}}

    # Baseline with fake outward and INaL = 3 - BM4,
    ind_4 = ind_2.copy()
    ind_4['i_nal_multiplier'] = 3
    dat_4, IC_4 = run_model([ind_4], 1)
    result_4 = rrc_search([ind_4], IC_4)
    all_data_4 = {**ind_4, **result_4, **{'dat':dat_4}}

    # Baseline with fake outward and fake inward - BM1
    ind_5 = ind_2.copy()
    ind_5['i_bias1_multiplier'] = -0.8
    dat_5, IC_5 = run_model([ind_5], 1)
    result_5 = rrc_search([ind_5], IC_5)
    all_data_5 = {**ind_5, **result_5, **{'dat':dat_5}}

    # Baseline Grandi Data - GBM
    base_ind_g = get_ind()
    base_ind_g['i_bias_multiplier'] = 0
    base_ind_g['i_bias1_multiplier'] = 0
    dat_g, IC_g = run_model([base_ind_g], 1, model = 'grandi_flat.mmt')
    result_g = rrc_search([base_ind_g], IC_g, model = 'grandi_flat.mmt')
    all_data_g = {**base_ind_g, **result_g, **{'dat':dat_g}}

    # Baseline Grandi + Outward - GBM2
    base_ind_g2 = base_ind_g.copy()
    base_ind_g2['i_bias_multiplier'] = 0.8
    base_ind_g2['i_bias1_multiplier'] = 0
    dat_g2, IC_g2 = run_model([base_ind_g2], 1, model = 'grandi_flat.mmt')
    result_g2 = rrc_search([base_ind_g2], IC_g2, model = 'grandi_flat.mmt')
    all_data_g2 = {**base_ind_g2, **result_g2, **{'dat':dat_g2}}

    # Baseline Grandi + Outward + ICaL = 3 - GBM3
    base_ind_g3 = base_ind_g2.copy()
    base_ind_g3['i_cal_pca_multiplier'] = 3
    dat_g3, IC_g3 = run_model([base_ind_g3], 1, model = 'grandi_flat.mmt')
    result_g3 = rrc_search([base_ind_g3], IC_g3, model = 'grandi_flat.mmt')
    all_data_g3 = {**base_ind_g3, **result_g3, **{'dat':dat_g3}}

    # Baseline Grandi with fake outward and fake inward - GBM1
    base_ind_g5  = base_ind_g2.copy()
    base_ind_g5['i_bias1_multiplier'] = -0.8
    dat_g5, IC_g5 = run_model([base_ind_g5], 1, model = 'grandi_flat.mmt')
    result_g5 = rrc_search([base_ind_g5], IC_g5, model = 'grandi_flat.mmt')
    all_data_g5 = {**base_ind_g5, **result_g5, **{'dat':dat_g5}}

    # Baseline Grandi + Outward + INaL = 1 - GBM4
    base_ind_g4 = base_ind_g2.copy()
    dat_g4, IC_g4 = run_model([base_ind_g4], 1, model = 'grandi_flat_NaL.mmt')
    result_g4 = rrc_search([base_ind_g4], IC_g4, model = 'grandi_flat_NaL.mmt')
    all_data_g4 = {**base_ind_g4, **result_g4, **{'dat':dat_g4}}

    ##########################################################################################################################################################
    # SAVE DATA
    pickle.dump({'BM':all_data, 'OM':all_data_o, 'OM1':all_data_0, 'OM2':all_data_1, 'OM3':all_data_0a, 'BM2':all_data_2, 'BM3':all_data_3, 'BM4':all_data_4, 'BM1':all_data_5, 'GBM':all_data_g, 'GBM2':all_data_g2, 'GBM3':all_data_g3, 'GBM4':all_data_g4, 'GBM1':all_data_g5}, open(save_to, 'wb'))

def get_robust_data(best_data_path = './data/best_data.csv', save_to = './data/robust_data.pkl'):
    best_data = pd.read_csv(best_data_path)
    ical_data = check_robustness(best_data, 'i_cal_pca_multiplier', [1, 2, 3, 4, 5, 6, 7, 8])
    ikr_kb_data = check_robustness(best_data, 'i_kr_multiplier', [1, 0.8, 0.6, 0.4, 0.2, 0], i_kb = 0.6)
    ical_data.extend(ikr_kb_data)
    ikr_data = check_robustness(best_data, 'i_kr_multiplier', [1, 0.8, 0.6, 0.4, 0.2, 0])
    ical_data.extend(ikr_data)
    robust_df = pd.DataFrame(ical_data)

    pickle.dump(robust_df, open(save_to, 'wb'))

def get_rrc_data(best_data_path = './data/best_data.csv', save_to = './data/rrc_data.csv'):
    """
    This function runs simulations to calculate the change in action potential duraction between the baseline ToR-ORd model and the 220 best GA individuals 
    at various stimuli. This data was used to produce Figures 3D and 3E. The data is stored as a csv file contained in the data/ folder and named rrc_data.csv.
    
    """
    # LOAD DATA
    best_data = pd.read_csv(best_data_path) 
    best_conds = best_data.filter(like = 'multiplier')
    best_conds.loc[220] = [1]*9 # add baseline as last row
    best_conds = best_conds.sort_index().reset_index(drop=True)

    ##########################################################################################################################################################
    # RUN SIMULATION
    print(time.time())
    time1 = time.time()

    if __name__ == "__main__":

        index = len(best_conds['i_cal_pca_multiplier'])
        args = [(i, best_conds, [0, 0.05, 0.1, 0.15, 0.2]) for i in range(index)]

        p = Pool() #allocates for the maximum amount of processers on laptop
        result = p.map(collect_rrc_data, args) 
        p.close()
        p.join()

    time2 = time.time()
    print('processing time: ', (time2-time1)/60, ' Minutes')
    print(time.time())

    ##########################################################################################################################################################
    # SAVE DATA
    df_data = pd.DataFrame(result)
    df_data.to_csv(save_to, index = False)


    
# Run functions to generate specific data files - it takes about 20 minutes to run all
get_baseline_torord_data(save_to= './data/baseline_torord_data_test.csv')
get_cond_data(best_data_path = './data/best_data.csv', save_to = './data/cond_data_test.pkl')
get_robust_data(best_data_path = './data/best_data.csv', save_to = './data/robust_data_test.pkl')

