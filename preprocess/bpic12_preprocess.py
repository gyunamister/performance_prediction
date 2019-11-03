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
	path = '../sample_data/BPI_Challenge_2012.csv'
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('Case ID')
	eventlog = eventlog.assign_activity('Activity')
	#eventlog = eventlog.assign_resource('Resource')
	eventlog['transition'] = eventlog['lifecycle:transition']
	eventlog['CompleteTimestamp'] = eventlog['Complete Timestamp']
	eventlog['CompleteTimestamp'] = eventlog['CompleteTimestamp'].apply(remove_micro_seconds)
	eventlog['CompleteTimestamp'] = eventlog['CompleteTimestamp'].str.replace('.', '/', regex=False)
	eventlog['Amount'] = eventlog['(case) AMOUNT_REQ']
	#eventlog = eventlog.assign_timestamp('Start Timestamp', name='Timestamp', format = '%Y/%m/%d %H:%M:%S')
	eventlog = eventlog.loc[(eventlog['Activity'].str.contains('W_', regex=False)) & ~(eventlog['Activity'].str.contains('SCHEDULE'))]
	caseid = ''
	temp_caseid=0
	start = False
	table = list()
	for row in eventlog.itertuples():
		if caseid != row.CASE_ID:
			caseid = row.CASE_ID
			temp_caseid += 1
			start = False
		if row.transition == 'START':
			if start == True:
				table.append(data)
			start = True
			data = list()
			data += [temp_caseid, row.Activity, row.Resource, row.CompleteTimestamp, row.Amount, '']
		if row.transition == 'COMPLETE':
			if start==True:
				data[-1] = row.CompleteTimestamp
			else:
				data = list()
				data += [temp_caseid, row.Activity, row.Resource, '', row.Amount, row.CompleteTimestamp]
			start = False
			table.append(data)
	headers = ['CASE_ID', 'Activity', 'Resource', 'StartTimestamp', 'Amount','CompleteTimestamp']
	df = pd.DataFrame(table, columns=headers)
	eventlog = Eventlog(df)
	eventlog = eventlog.assign_timestamp(name='StartTimestamp', new_name='StartTimestamp', _format = '%Y.%m.%d %H:%M:%S', errors='raise')
	eventlog = eventlog.assign_timestamp(name='CompleteTimestamp', new_name='CompleteTimestamp', _format = '%Y/%m/%d %H:%M:%S', errors='raise')
	eventlog['Duration'] = (eventlog['CompleteTimestamp'] - eventlog['StartTimestamp']).apply(to_minute)

	eventlog.dropna(subset=['Resource', 'StartTimestamp', 'CompleteTimestamp'],inplace=True)
	eventlog.to_csv('../sample_data/BPIC2012.csv')