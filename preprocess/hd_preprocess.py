import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

p = Path(__file__).resolve().parents[2]
sys.path.append(os.path.abspath(str(p)))
from PyProM.src.data.Eventlog import Eventlog

if __name__=='__main__':
	path = '../sample_data/helpdesk2.csv'
	eventlog = Eventlog.from_txt(path, sep=',')
	eventlog = eventlog.assign_caseid('CASEOID')
	eventlog = eventlog.assign_activity('ACTIVITYOID')
	eventlog = eventlog.assign_timestamp('TIMESTAMP')
	eventlog = eventlog.loc[(eventlog['TIMESTAMP']>"01-01-2011") & (eventlog['TIMESTAMP']<"06-30-2012")]
	print(eventlog)
	eventlog.to_csv('../sample_data/HD1.csv')