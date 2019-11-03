import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))
from PyProM.src.data.Eventlog import Eventlog

def remove_micro_seconds(x):
	if len(x) > 19:
		x = x[:19]
	return x

def to_minute(x):
	if np.isnan(x.seconds):
		return x
	#return int(x.seconds/60)
	return math.ceil(x.seconds/60)

if __name__=='__main__':
	path = '../sample_data/BPIC15_1_unfiltered.csv'
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('CASEOID')
	eventlog = eventlog.assign_activity('ACTIVITYOID')
	eventlog = eventlog.assign_timestamp('ENDAT')
	#eventlog = eventlog.assign_resource('Resource')
	#eventlog['transition'] = eventlog['lifecycle:transition']
	import datetime
	date_after = datetime.date(2011, 1, 1)
	date_before = datetime.date(2014, 12, 31)
	eventlog = eventlog.loc[(eventlog['TIMESTAMP']>date_after) & (eventlog['TIMESTAMP']<date_before)]
	print(eventlog)
	eventlog.to_csv('../sample_data/BPIC15_1.csv')