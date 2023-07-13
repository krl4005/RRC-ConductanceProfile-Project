from important_functions import run_model
import pandas as pd

dat, IC = run_model(None, 1, path = './models/')
baseline_torord = pd.DataFrame({'t': dat['engine.time'], 'v': dat['membrane.v']})
baseline_torord.to_csv('./data/baseline_torord_data.csv', index = False)

