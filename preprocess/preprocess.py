from PyProM.src.data.Eventlog import Eventlog
import pandas as pd

class Preprocessor(object):
    def load_eventlog(self, path, case, activity, timestamp, encoding=None, clear=True):
        eventlog = pd.read_csv(path, sep = ',', engine='python', encoding=encoding)
        eventlog.sort_values([case, timestamp], inplace=True)
        eventlog.reset_index(drop=True, inplace=True)
        eventlog = Eventlog(eventlog)
        eventlog = eventlog.assign_caseid(case)
        eventlog = eventlog.assign_activity(activity)
        eventlog = eventlog.assign_timestamp(timestamp)
        if clear == True:
            eventlog = eventlog.clear_columns()
        return eventlog