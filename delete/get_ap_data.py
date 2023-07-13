# IMPORT FIGURES
import pickle
import pandas as pd
from important_functions import get_ind, run_model, rrc_search

##########################################################################################################################################################
# LOAD DATA
all_trials = pd.read_csv('./data/all_data.csv.gz')
best_data = pd.read_csv('./data/best_data.csv')

##########################################################################################################################################################
# RUN SIMULATIONS

# Baseline Data - BM
base_ind = get_ind()
base_ind['i_bias_multiplier'] = 0
base_ind['i_bias1_multiplier'] = 0
dat, IC = run_model([base_ind], 1, path='./models/')
result = rrc_search([base_ind], IC, path = './models/')
all_data = {**base_ind, **result, **{'dat':dat}}

# Optimized Data - OM
ind = best_data.iloc[0].filter(like = 'multiplier').to_dict()
ind['i_bias_multiplier'] = 0
ind['i_bias1_multiplier'] = 0
dat_o, IC_o = run_model([ind], 1, path='./models/')
result_o = rrc_search([ind], IC_o, path = './models/')
all_data_o = {**ind, **result_o, **{'dat':dat_o}}

# Optimized Data without INaL - OM1
ind_0 = ind.copy()
ind_0['i_nal_multiplier'] = 0
dat_0, IC_0 = run_model([ind_0], 1, path='./models/')
result_0 = rrc_search([ind_0], IC_0, path = './models/')
all_data_0 = {**ind_0, **result_0, **{'dat':dat_0}}

# Optimized Data without INaL and ICaL Rescue - OM2
ind_1 = ind_0.copy()
ind_1['i_cal_pca_multiplier'] = 3
dat_1, IC_1 = run_model([ind_1], 1, path='./models/')
result_1 = rrc_search([ind_1], IC_1, path = './models/')
all_data_1 = {**ind_1, **result_1, **{'dat':dat_1}}

# Optimized Data without ICaL - OM3
ind_0a = ind.copy()
ind_0a['i_cal_pca_multiplier'] = 0
dat_0a, IC_0a = run_model([ind_0a], 1, path='./models/')
result_0a = rrc_search([ind_0a], IC_0a, path = './models/')
all_data_0a = {**ind_0a, **result_0a, **{'dat':dat_0a}}

# Baseline with fake outward - BM2
ind_2 = base_ind.copy()
ind_2['i_bias_multiplier'] = 0.8
ind_2['i_bias1_multiplier'] = 0
dat_2, IC_2 = run_model([ind_2], 1, path='./models/')
result_2 = rrc_search([ind_2], IC_2, path = './models/')
all_data_2 = {**ind_2, **result_2, **{'dat':dat_2}}

# Baseline with fake outward and ICaL = 3 - BM3
ind_3 = ind_2.copy()
ind_3['i_cal_pca_multiplier'] = 3
dat_3, IC_3 = run_model([ind_3], 1, path='./models/')
result_3 = rrc_search([ind_3], IC_3, path = './models/')
all_data_3 = {**ind_3, **result_3, **{'dat':dat_3}}

# Baseline with fake outward and INaL = 3 - BM4,
ind_4 = ind_2.copy()
ind_4['i_nal_multiplier'] = 3
dat_4, IC_4 = run_model([ind_4], 1, path='./models/')
result_4 = rrc_search([ind_4], IC_4, path = './models/')
all_data_4 = {**ind_4, **result_4, **{'dat':dat_4}}

# Baseline with fake outward and fake inward - BM1
ind_5 = ind_2.copy()
ind_5['i_bias1_multiplier'] = -0.8
dat_5, IC_5 = run_model([ind_5], 1, path='./models/')
result_5 = rrc_search([ind_5], IC_5, path = './models/')
all_data_5 = {**ind_5, **result_5, **{'dat':dat_5}}

# Baseline Grandi Data - GBM
base_ind_g = get_ind()
base_ind_g['i_bias_multiplier'] = 0
base_ind_g['i_bias1_multiplier'] = 0
dat_g, IC_g = run_model([base_ind_g], 1, path='./models/', model = 'grandi_flat.mmt')
result_g = rrc_search([base_ind_g], IC_g, path='./models/', model = 'grandi_flat.mmt')
all_data_g = {**base_ind_g, **result_g, **{'dat':dat_g}}

# Baseline Grandi + Outward - GBM2
base_ind_g2 = base_ind_g.copy()
base_ind_g2['i_bias_multiplier'] = 0.8
base_ind_g2['i_bias1_multiplier'] = 0
dat_g2, IC_g2 = run_model([base_ind_g2], 1, path='./models/', model = 'grandi_flat.mmt')
result_g2 = rrc_search([base_ind_g2], IC_g2, path='./models/', model = 'grandi_flat.mmt')
all_data_g2 = {**base_ind_g2, **result_g2, **{'dat':dat_g2}}

# Baseline Grandi + Outward + ICaL = 3 - GBM3
base_ind_g3 = base_ind_g2.copy()
base_ind_g3['i_cal_pca_multiplier'] = 3
dat_g3, IC_g3 = run_model([base_ind_g3], 1, path='./models/', model = 'grandi_flat.mmt')
result_g3 = rrc_search([base_ind_g3], IC_g3, path='./models/', model = 'grandi_flat.mmt')
all_data_g3 = {**base_ind_g3, **result_g3, **{'dat':dat_g3}}

# Baseline Grandi with fake outward and fake inward - GBM1
base_ind_g5  = base_ind_g2.copy()
base_ind_g5['i_bias1_multiplier'] = -0.8
dat_g5, IC_g5 = run_model([base_ind_g5], 1, path='./models/', model = 'grandi_flat.mmt')
result_g5 = rrc_search([base_ind_g5], IC_g5, path='./models/', model = 'grandi_flat.mmt')
all_data_g5 = {**base_ind_g5, **result_g5, **{'dat':dat_g5}}

# Baseline Grandi + Outward + INaL = 1 - GBM4
base_ind_g4 = base_ind_g2.copy()
dat_g4, IC_g4 = run_model([base_ind_g4], 1, path='./models/', model = 'grandi_flat_NaL.mmt')
result_g4 = rrc_search([base_ind_g4], IC_g4, path='./models/', model = 'grandi_flat_NaL.mmt')
all_data_g4 = {**base_ind_g4, **result_g4, **{'dat':dat_g4}}

##########################################################################################################################################################
# SAVE DATA
pickle.dump({'BM':all_data, 'OM':all_data_o, 'OM1':all_data_0, 'OM2':all_data_1, 'OM3':all_data_0a, 'BM2':all_data_2, 'BM3':all_data_3, 'BM4':all_data_4, 'BM1':all_data_5, 'GBM':all_data_g, 'GBM2':all_data_g2, 'GBM3':all_data_g3, 'GBM4':all_data_g4, 'GBM1':all_data_g5}, open('./data/cond_data.pkl', 'wb'))


# %%
