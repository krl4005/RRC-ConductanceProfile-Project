# IMPORT FUNCTIONS
import pandas as pd
from multiprocessing import Pool
import time
from important_functions import calc_APD, run_model

##########################################################################################################################################################
# DEFINE FUNCTIONS
def collect_rrc_data(args):
    i, best_conds, stims = args
    ind_data = {}
    ind = best_conds.filter(like= 'multiplier').iloc[i].to_dict()
    for s in list(range(0, len(stims))):
        dat, IC = run_model([ind], 1, stim = 5.3, stim_1 = stims[s], start = 0.1, start_1 = 4, length = 1, length_1 = 996, cl = 1000, prepace = 600, I0 = 0, path = './models/', model = 'tor_ord_endo2.mmt') 
        ind_data['t_'+str(stims[s])] = [dat['engine.time'].tolist()]
        ind_data['v_'+str(stims[s])] = [dat['membrane.v'].tolist()]
        ind_data['apd_'+str(stims[s])] = calc_APD(dat['engine.time'], dat['membrane.v'], 90)
        if s == 0:
            base_apd = calc_APD(dat['engine.time'], dat['membrane.v'], 90)
        ind_data['delapd_'+str(stims[s])] = calc_APD(dat['engine.time'], dat['membrane.v'], 90) - base_apd
        
    data = {**ind, **ind_data}
    return data

##########################################################################################################################################################
# LOAD DATA
best_data = pd.read_csv('./data/best_data.csv') 
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
df_data.to_csv('./data/rrc_data.csv', index = False)

