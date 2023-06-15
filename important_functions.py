#%%
# IMPORT FUNCTIONS
import myokit
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from scipy import stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# DEFINE FUNCTIONS
def get_ind(vals = [1,1,1,1,1,1,1,1,1,1], celltype = 'adult'):
    if celltype == 'ipsc':
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_f_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    else:
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    return(ind)

def run_model(ind, beats, stim = 5.3, stim_1 = 0, start = 0.1, start_1 = 0, length = 1, length_1 = 0, cl = 1000, prepace = 600, I0 = 0, path = '../', model = 'tor_ord_endo2.mmt'): 
    mod, proto = get_ind_data(ind, path, model = model)
    proto.schedule(stim, start, length, cl, 0) 
    if stim_1 != 0:
        proto.schedule(stim_1, start_1, length_1, cl, 1)
    sim = myokit.Simulation(mod,proto)

    if I0 != 0:
        sim.set_state(I0)

    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    return(dat, IC) 

def rrc_search(ind, IC, path = '../', model = 'tor_ord_endo2.mmt'):
    all_data = []
    APs = list(range(10004, 100004, 5000))

    mod, proto = get_ind_data(ind, path, model) 
    if model == 'kernik.mmt' or model == 'kernik_INaL.mmt':
        proto.schedule(1, 0.2, 5, 1000, 0)
    else:
        proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.3, 5004, 996, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    dat = sim.run(7000)

    d0 = get_last_ap(dat, 4)
    result_abnormal0 = detect_abnormal_ap(d0['t'], d0['v'])
    all_data.append({**{'t_rrc': d0['t'], 'v_rrc': d0['v'], 'stim': 0}, **result_abnormal0})

    d3 = get_last_ap(dat, 5)
    result_abnormal3 = detect_abnormal_ap(d3['t'], d3['t'])
    all_data.append({**{'t_rrc': d3['t'], 'v_rrc': d3['v'], 'stim': 0.3}, **result_abnormal3})

    if result_abnormal0['result'] == 1:
        RRC = 0

    elif result_abnormal3['result'] == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        low = 0
        high = 0.3
        for i in list(range(0,len(APs))):
            mid = round((low + (high-low)/2), 4) 

            sim.reset()
            sim.set_state(IC)
            proto.schedule(mid, APs[i], 996, 1000, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+2000)

            data = get_last_ap(dat, int((APs[i]-4)/1000))
            result_abnormal = detect_abnormal_ap(data['t'], data['v'])
            all_data.append({**{'t_rrc': data['t'], 'v_rrc': data['v'], 'stim': mid}, **result_abnormal})
            
            if result_abnormal['result'] == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid

            if (high-low)<0.0025: 
                break 
        
        for i in list(range(1, len(all_data))):
            if all_data[-i]['result'] == 0:
                RRC = all_data[-i]['stim']
                break
            else:
                RRC = 0 #in this case there would be no stim without an RA

    result = {'RRC':RRC, 'data':all_data}

    return(result)

def get_last_ap(dat, AP, cl = 1000, type = 'full'):

    if type == 'full':
        start_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), AP*cl))
        end_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), (AP+1)*cl))

        t = np.array(dat['engine.time'][start_ap:end_ap])
        t = t-t[0]
        v = np.array(dat['membrane.v'][start_ap:end_ap])
        cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
        i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v
        data['cai'] = cai
        data['i_ion'] = i_ion
    
    else:
        # Get t, v, and cai for second to last AP#######################
        ti, vol = dat

        start_ap = list(ti).index(closest(ti, AP*cl))
        end_ap = list(ti).index(closest(ti, (AP+1)*cl))

        t = np.array(ti[start_ap:end_ap])
        v = np.array(vol[start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v

    return (data)

def check_physio(ap_features, cost = 'function_2', feature_targets = {'Vm_peak': [10, 33, 55], 'dvdt_max': [100, 347, 1000], 'apd40': [85, 198, 320], 'apd50': [110, 220, 430], 'apd90': [180, 271, 440], 'triangulation': [50, 73, 150], 'RMP': [-95, -88, -80]}):

    error = 0
    if cost == 'function_1':
        for k, v in feature_targets.items():
            if ((ap_features[k] > v[0]) and (ap_features[k] < v[2])):
                error+=0
            else:
                error+=(v[1]-ap_features[k])**2
    else:
        for k, v in feature_targets.items():
            if ((ap_features[k] < v[0]) and (ap_features[k] > v[2])):
                error+=1000

    return(error)

def get_rrc_error(RRC, cost):

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = 0
    RRC_est = RRC

    if cost == 'function_1':
        error += round((0.3 - (np.abs(RRC)))*20000)

    else:
        # This just returns the error from the first RRC protocol
        stims = np.asarray([0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])
        pos_error = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0]
        i = (np.abs(stims - RRC)).argmin()
        check_low = stims[i]-stims[i-1]
        check_high = stims[i]-stims[i+1]

        if check_high<check_low:
            RRC_est = stims[i-1]
            error = pos_error[i-1]
        else:
            RRC_est = i
            error += pos_error[i]

    return error, RRC_est

def get_features(t,v,cai=None):

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))

    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    if cai is not None: 
        # Calcium/CaT features######################## 
        max_cai = np.max(cai)
        max_cai_idx = np.argmax(cai)
        max_cai_time = t[max_cai_idx]
        cat_amp = np.max(cai) - np.min(cai)
        ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small
        ap_features['cat_peak'] = max_cai_time

        for cat_pct in [90]:
            cat_recov = max_cai - cat_amp * cat_pct / 100
            idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
            catd_val = t[idx_catd+max_cai_idx]

            ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def check_physio_torord(t, v, path = '../', filter = 'no'):

    # Cut off the upstroke of the AP for profile
    t_ind = list(t[150:len(t)]) 
    v_ind = list(v[150:len(t)])

    # Baseline tor-ord model & cut off upstroke
    base_df = pd.read_csv('./data/baseline_torord_data.csv')
    t_base = list(base_df['t'])[150:len(t)]
    v_base = list(base_df['v'])[150:len(t)]

    # Cut off the upstroke of the AP for the tor-ord data
    if filter == 'no':
        time, vol_10, vol_90 = get_torord_phys_data()
        t = time[150:len(time)]
        v_10 = vol_10[150:len(time)]
        v_90 = vol_90[150:len(time)]

    else:
        t, v_10, v_90 = get_torord_phys_data()

    result = 0 # valid AP
    error = 0
    check_times = []
    data = {}

    for i in list(range(0, len(t_ind))):
        t_dat = closest(t, t_ind[i]) # find the value closest to the ind's time within the exp data time list
        t_dat_base = closest(t_base, t_ind[i])
        t_dat_i = np.where(np.array(t)==t_dat)[0][0] #find the index of the closest value in the list 
        t_dat_base_i = np.where(np.array(t_base)==t_dat_base)[0][0] #find the index of the closest value in the list 
        v_model = v_ind[i]
        v_lowerbound = v_10[t_dat_i]
        v_upperbound = v_90[t_dat_i]
        v_torord = v_base[t_dat_base_i] 

        check_times.append(np.abs(t_ind[i] - t_dat))

        if v_model < v_lowerbound or v_model > v_upperbound:
            result = 1 # not a valid AP
            error += (v_model - v_torord)**2
    
    data['result'] = result
    data['error'] = error
    data['check_times'] = check_times

    return(data)

def get_torord_phys_data():
    data = pd.read_csv('./data/APbounds.csv')
    time = [x - 9.1666666669999994 for x in list(data['t'])] #shift action potential to match solutions
    t = time[275:len(data['v_10'])]
    v_10 = list(data['v_10'])[275:len(data['v_10'])]
    v_90 = list(data['v_90'])[275:len(data['v_10'])]

    data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
    data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})
    
    # FILTER V_10
    v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
    v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
    v_10_new = v_10_new.dropna()
    v_10 = list(v_10_start) + list(v_10_new)
    t = list(data_start['t']) + list(data['t'])

    # FILTER V_90
    v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
    v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
    v_90_new = v_90_new.dropna()
    v_90 = list(v_90_start) + list(v_90_new)


    return(t, v_10, v_90)

def calc_APD(t, v, apd_pct):
    t = np.array(t)
    v = np.array(v)
    t = [i-t[0] for i in t]
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]

    result = detect_abnormal_ap(t, v)
    if len(result['RMP']) == 0:
        apd_val = max(t)

    return(apd_val) 

def get_ind_data(ind, path = '../', model = 'tor_ord_endo2.mmt'):
    mod, proto, x = myokit.load(path+model)
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def detect_abnormal_ap(t, v):

    slopes = []
    for i in list(range(0, len(v)-1)):
        if t[i] > 100 and v[i] < 20:
            m = (v[i+1]-v[i])/(t[i+1]-t[i])
            slopes.append(round(m, 2))
        else:
            slopes.append(-2.0)

    # EAD CODE
    rises_idx = np.where(np.array(slopes)>0)
    rises_groups = []
    for k, g in groupby(enumerate(rises_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rises_groups.append(list(map(itemgetter(1), g)))

    # RF CODE
    rpm_idx = np.where(np.array(slopes) == 0)
    rpm_groups = []
    for k, g in groupby(enumerate(rpm_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rpm_groups.append(list(map(itemgetter(1), g)))

    flat_groups = [group for group in rpm_groups if v[group[-1]]<-70]

    # CHECK PHASE 4 RF
    if len(flat_groups)>0:
        RMP_start = flat_groups[0][0]
        v_rm = v[RMP_start:len(v)]
        t_rm = t[RMP_start:len(v)]
        slope = (v_rm[-1]-v_rm[0])/(t_rm[-1]-t_rm[0])
        if slope < 0.01:
            for group in rises_groups:
                if v[group[0]]<-70:
                    rises_groups.remove(group)


    if len(flat_groups)>0 and len(rises_groups)==0:
        info = "normal AP" 
        result = 0
    else:
        info = "abnormal AP"
        result = 1

    data = {'info': info, 'result':result, 'EADs':rises_groups, 'RMP':flat_groups}

    return(data)

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def add_scalebar(axs, section, y_pos = -0.1):
    # FORMAT X AXIS
    if section == 0:
        xmin, xmax, ymin, ymax = axs.axis()
        scalebar = AnchoredSizeBar(axs.transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,y_pos), bbox_transform =axs.transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
        axs.add_artist(scalebar)
        axs.spines[['bottom']].set_visible(False)
        axs.tick_params(bottom=False)
        axs.tick_params(labelbottom=False)
    else:
        for i in list(range(0, len(section))):
            xmin, xmax, ymin, ymax = axs[section[i][0], section[i][1]].axis()
            scalebar = AnchoredSizeBar(axs[section[i][0], section[i][1]].transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,-0.2), bbox_transform =axs[section[i][0], section[i][1]].transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
            axs[section[i][0], section[i][1]].add_artist(scalebar)
            axs[section[i][0], section[i][1]].spines[['bottom']].set_visible(False)
            axs[section[i][0], section[i][1]].tick_params(bottom=False)
            axs[section[i][0], section[i][1]].tick_params(labelbottom=False)

def new_parameter_convergence(all_trials, fitness='fitness'):
    all_dicts = []

    
    for t in list(range(0, max(all_trials['trial']))):
        old_data = all_trials[(all_trials['trial']==t) & (all_trials['gen']==0)].sort_values(fitness).iloc[0:100]
        for g in list(range(0, max(all_trials[(all_trials['trial']==t)]['gen']))):
            data = all_trials[(all_trials['trial']==t) & (all_trials['gen']==g)].sort_values(fitness).iloc[0:100]
            data = pd.concat([old_data, data])
            data = data.drop_duplicates(subset=data.filter(like='multiplier').columns.to_list())
            data_var = data.sort_values(fitness).iloc[0:100].filter(like = 'multiplier').var().to_dict()
            data_var['generation'] = g
            data_var['trial'] = t
            all_dicts.append(data_var)
            old_data = data

    df_dicts = pd.DataFrame(all_dicts)

    average_dicts = []
    for g in list(range(0, max(df_dicts['generation']))):
        average_dicts.append(df_dicts[df_dicts['generation']==g].mean().to_dict())
    df_dicts_average = pd.DataFrame(average_dicts)
    
    return df_dicts_average

def get_sensitivities(all_trials, error):
    population = all_trials[all_trials['gen']==0].sort_values(error) 
    good_pop = population.iloc[0:160] #top 10% of the population
    bad_pop = population.iloc[160:len(population['gen'])] #remaining 90% of the population

    sensitivities = []
    pvalues = []
    for cond in good_pop.filter(like='multiplier').columns.to_list():
        stat, pvalue = stats.ks_2samp(good_pop[cond], bad_pop[cond])
        sensitivities.append(stat)
        pvalues.append(pvalue)
    return sensitivities, pvalues

# %%
