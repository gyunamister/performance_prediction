import numpy as np
import datetime
import copy
from PyProM.src.mining.transition_matrix import TransitionMatrix
from PyProM.src.data.Eventlog import Eventlog



def get_dt_from_string(dt):
    """
    If the date is expressed as string, do the conversion to a datetime.datetime object

    Parameters
    -----------
    dt
        Date (string or datetime.datetime)

    Returns
    -----------
    dt
        Datetime object
    """
    if type(dt) is str:
        return datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def filter_log_by_timestamp(log, start_time, complete_time, dt1, dt2):
    dt1 = get_dt_from_string(dt1)
    dt2 = get_dt_from_string(dt2)
    #start, complete이 모두 있는 경우

    #complete만 있는 경우
    #처리량이 많으므로 1차적으로 filtering 수행
    log = log.loc[log['CASE_ID'].isin(log.loc[(log[complete_time]>dt1) & (log[complete_time]<dt2), 'CASE_ID'])]
    log.reset_index(inplace=True)
    #log.to_csv("./result/log.csv")

    maxlen = log.count_event()
    valid_idx_list = list()
    for row in log.itertuples():
        if row.Index+1 < maxlen:
            next_caseid = log['CASE_ID'][row.Index+1]
        else:
            next_caseid = ''
        if row.TIMESTAMP > dt1 and row.TIMESTAMP <dt2:
            if row.CASE_ID == next_caseid:
                valid_idx_list.append(row.Index)
                if row.Index+1 < maxlen:
                    #add next to calculate sojourn time
                    valid_idx_list.append(row.Index+1)
            else:
                valid_idx_list.append(row.Index)
    valid_idx_list = list(set(valid_idx_list))
    log = log.loc[log.index.isin(valid_idx_list)]
    #log.to_csv("./result/filtered_log.csv")
    #import sys
    #sys.exit("Done")
    return log

def filter_log_by_timestamp_with_start(log, start_time, complete_time, dt1, dt2, measure, agg):
    dt1 = get_dt_from_string(dt1)
    dt2 = get_dt_from_string(dt2)
    #start, complete이 모두 있는 경우

    #complete만 있는 경우
    #처리량이 많으므로 1차적으로 filtering 수행
    valid_idx_list = list()
    complete_on = log.loc[(log[start_time]<dt1) & (log[complete_time]>dt1)].index
    valid_idx_list += list(complete_on)
    start_on = log.loc[(log[start_time]<dt2) & (log[complete_time]>dt2)].index
    valid_idx_list += list(start_on)
    both_on = log.loc[(log[start_time]>dt1) & (log[complete_time]<dt2)].index
    valid_idx_list += list(both_on)
    filt_log = log.loc[log.index.isin(valid_idx_list)]
    # length를 구할때는 window에 걸치는 경우 경계 지점의 시간으로 measure 계산해야함
    if measure=='processing':
        valid_idx_list = list()
        start_on = log.loc[(log[start_time]<dt2) & (log[complete_time]>dt2)].index
        valid_idx_list += list(start_on)
        both_on = log.loc[(log[start_time]>dt1) & (log[complete_time]<dt2)].index
        valid_idx_list += list(both_on)

        filt_log = log.loc[log.index.isin(valid_idx_list)]

    elif measure =='waiting':
        """
        valid_idx_list = list()
        complete_on = log.loc[(log[start_time]<dt1) & (log[complete_time]>dt1)].index
        valid_idx_list += list(complete_on)
        both_on = log.loc[(log[start_time]>dt1) & (log[complete_time]<dt2)].index
        valid_idx_list += list(both_on)
        filt_log = log.loc[log.index.isin(valid_idx_list)]

        prev_caseid = ''
        filt_index = 0
        for row in filt_log.itertuples():
            if prev_caseid != row.CASE_ID:
                prev_caseid = row.CASE_ID
                continue
            else:
                next_caseid = filt_log.iloc[filt_index]['CASE_ID']
                if next_caseid != row.CASE_ID:
                    if next_log_caseid == log.loc[row.Index, 'CASE_ID']:
        """
        valid_caseid = log.loc[((log[start_time]<dt1) & (log[complete_time]>dt1)) | ((log[start_time]>dt1) & (log[complete_time]<dt2)), 'CASE_ID']
        #valid_caseid = log.loc[log[complete_time]<dt2, 'CASE_ID']
        log = log.loc[log['CASE_ID'].isin(valid_caseid)]
        log.reset_index(inplace=True)
        #log.to_csv("./result/log.csv")

        maxlen = log.count_event()
        valid_idx_list = list()
        for row in log.itertuples():
            if row.Index+1 < maxlen:
                next_caseid = log.loc[row.Index+1, 'CASE_ID']
            else:
                next_caseid = ''
            if (row.START_TIMESTAMP<dt1 and row.TIMESTAMP>dt1) or (row.START_TIMESTAMP>dt1 and row.TIMESTAMP<dt2):
                if row.CASE_ID == next_caseid:
                    valid_idx_list.append(row.Index)
                    if row.Index+1 < maxlen:
                        #add next to calculate sojourn time
                        valid_idx_list.append(row.Index+1)
                else:
                    valid_idx_list.append(row.Index)
        valid_idx_list = list(set(valid_idx_list))
        filt_log = log.loc[log.index.isin(valid_idx_list)]
        filt_log.to_csv("./result/filt_log_1.csv")


    """
    if agg=='length':
        add_rows = list()
        prev_caseid = ''
        #
        filt_index = 0
        for row in filt_log.itertuples():
            #이전 row와 caseid가 다르면 첫지점에 dummy 추가
            log_index = row.Index
            if prev_caseid != row.CASE_ID:
                if filt_index >= 1:
                    prev_row = log.loc[log_index-1]
                    if row.CASE_ID == prev_row['CASE_ID']:
                        prev_row[start_time] = dt1
                        prev_row[complete_time] = dt1
                        add_rows.append(prev_row)
                if filt_index <= len(filt_log) - 2:
                    if row.CASE_ID != filt_log.iloc[filt_index+1]['CASE_ID']:
                        next_row = log.loc[log_index+1]
                        if log_index <= len(log) - 2:
                            if row.CASE_ID == next_row['CASE_ID']:
                                next_row[start_time] = dt2
                                next_row[complete_time] = dt2
                                add_rows.append(next_row)
                prev_caseid = row.CASE_ID
            #이전 row와 caseid가 같으면 끝지점에 dummy 추가
            else:
                if filt_index <= len(filt_log) - 2:
                    if row.CASE_ID != filt_log.iloc[filt_index+1]['CASE_ID']:
                        next_row = log.loc[log_index+1]
                        if log_index <= len(log) - 2:
                            if row.CASE_ID == next_row['CASE_ID']:
                                next_row[start_time] = dt2
                                next_row[complete_time] = dt2
                                add_rows.append(next_row)
                prev_caseid = row.CASE_ID
            filt_index += 1
        #filt_log.to_csv("./result/filt_log_1.csv")
        filt_log = filt_log.append(add_rows)
        filt_log.sort_index(inplace=True)
        filt_log = Eventlog(filt_log)

        #1. 이전 Complete이 현재 Start보다 늦으면 현재 Start를 이전 Complete으로 대체하고 이전 row 삭제, (데이터 오류로 발생하는 경우가 있으므로) 이전 Start와 이전 Complete이 같은 경우(즉, Dummy)만 삭제
        #2. 다음 Start가 현재 Complete보다 빠르면 현재 Complete을 다음 Start로 대체하고 다음 row 삭제, (데이터 오류로 발생하는 경우가 있으므로) 다음 Start와 다음 Complete이 같은 경우(즉, Dummy)만 삭제
        prev_caseid = ''
        remove_list = list()
        #
        filt_index = 0
        for row in filt_log.itertuples():
            if prev_caseid != row.CASE_ID:
                if row.START_TIMESTAMP == row.TIMESTAMP:
                    if row.CASE_ID == filt_log.iloc[filt_index+1]['CASE_ID']:
                        next_start = filt_log.iloc[filt_index+1][start_time]
                        if next_start < row.START_TIMESTAMP:
                            filt_log.iloc[filt_index+1][start_time] = row.START_TIMESTAMP
                            remove_list.append(row.Index)
                prev_caseid = row.CASE_ID
            else:
                if row.START_TIMESTAMP == row.TIMESTAMP:
                    if row.CASE_ID == filt_log.iloc[filt_index-1]['CASE_ID']:
                        prev_complete = filt_log.iloc[filt_index-1][complete_time]
                        if prev_complete > row.TIMESTAMP:
                            filt_log.iloc[filt_index-1][complete_time] = row.TIMESTAMP
                            remove_list.append(row.Index)
            filt_index += 1
        filt_log = filt_log.loc[~filt_log.index.isin(remove_list)]


        """



        #filt_log.to_csv("./result/filt_log_2.csv")
    return filt_log

def convert_to_hex(rgba_color):
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)



def get_string_from_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def apply_edge_threshold(transition_matrix, edge_threshold):
    transition_matrix = copy.deepcopy(transition_matrix)
    for ai in list(transition_matrix.keys()):
        for aj in list(transition_matrix[ai]['outgoings'].keys()):
            if transition_matrix[ai]['outgoings'][aj]['edge_count'] < edge_threshold:
                del transition_matrix[ai]['outgoings'][aj]
    return transition_matrix

def apply_node_threshold(transition_matrix, node_threshold):
    transition_matrix = copy.deepcopy(transition_matrix)
    for ai in list(transition_matrix.keys()):
        if transition_matrix[ai]['node_count'] < node_threshold:
            del transition_matrix[ai]
    return transition_matrix

def compute_avg_queue_len(transition_matrix, measure, log_range):
    for ai in transition_matrix:
        if measure in transition_matrix[ai]:
            transition_matrix[ai]['length_{}'.format(measure)] = sum(transition_matrix[ai][measure], datetime.timedelta(0))/log_range
        else:
            transition_matrix[ai]['length_{}'.format(measure)] = 0
        for aj in transition_matrix[ai]['outgoings']:
            transition_matrix[ai]['outgoings'][aj]['length_{}'.format(measure)] = sum(transition_matrix[ai]['outgoings'][aj][measure], datetime.timedelta(0))/log_range
    return transition_matrix

def compute_std_time(transition_matrix, measure):
    for ai in transition_matrix:
        if measure in transition_matrix[ai] and len(transition_matrix[ai][measure])!=0:
            measure_list = [divmod(x.days * 86400 + x.seconds, 86400) for x in transition_matrix[ai][measure]]
            measure_list = [24*60*x[0] + x[1]/60 for x in measure_list]
            std = np.std(measure_list)
        else:
            std = 0
        """
        if avg_minute < 0:
            print(ai, avg_minute)
        """
        transition_matrix[ai]['std_{}'.format(measure)] = std
        for aj in transition_matrix[ai]['outgoings']:
            if len(transition_matrix[ai]['outgoings'][aj][measure])==0:
                std=0
            else:
                measure_list = [divmod(x.days * 86400 + x.seconds, 86400) for x in transition_matrix[ai]['outgoings'][aj][measure]]
                measure_list = [24*60*x[0] + x[1]/60 for x in measure_list]
                std = np.std(measure_list)
            transition_matrix[ai]['outgoings'][aj]['std_{}'.format(measure)] = std
    #print(transition_matrix)
    return transition_matrix

def compute_avg_time(transition_matrix, measure):
    from statistics import mean
    for ai in transition_matrix:
        if measure in transition_matrix[ai] and len(transition_matrix[ai][measure])!=0:
            #avg = sum(transition_matrix[ai][measure], datetime.timedelta(0))/len(transition_matrix[ai][measure])
            #avg = divmod(avg.days * 86400 + avg.seconds, 86400)
            #avg_minute = 24*60*avg[0] + avg[1]/60
            upper = np.percentile(transition_matrix[ai][measure], 50)
            #lower = np.percentile(transition_matrix[ai][measure], 0)
            lower=0
            transition_matrix[ai][measure] = [x for x in transition_matrix[ai][measure] if x<=upper and x>=lower]
            if len(transition_matrix[ai][measure])!=0:
                avg_minute = sum(transition_matrix[ai][measure])/len(transition_matrix[ai][measure])
            else:
                avg_minute = 0
        else:
            avg_minute = 0
        """
        if avg_minute < 0:
            print(ai, avg_minute)
        """
        transition_matrix[ai]['avg_{}'.format(measure)] = avg_minute
        for aj in transition_matrix[ai]['outgoings']:
            if len(transition_matrix[ai]['outgoings'][aj][measure])==0:
                avg_minute=0
            else:
                #avg = sum(transition_matrix[ai]['outgoings'][aj][measure], datetime.timedelta(0))/len(transition_matrix[ai]['outgoings'][aj][measure])
                #avg = divmod(avg.days * 86400 + avg.seconds, 86400)
                #avg_minute = 24*60*avg[0] + avg[1]/60
                avg_minute = np.mean(transition_matrix[ai]['outgoings'][aj][measure])
            transition_matrix[ai]['outgoings'][aj]['avg_{}'.format(measure)] = avg_minute
    #print(transition_matrix)
    return transition_matrix

def compute_cnt(transition_matrix, measure):
    for ai in transition_matrix:
        if measure in transition_matrix[ai]:
            transition_matrix[ai]['count_{}'.format(measure)] = len(transition_matrix[ai][measure])
        else:
            transition_matrix[ai]['count_{}'.format(measure)] = 0
        for aj in transition_matrix[ai]['outgoings']:
            transition_matrix[ai]['outgoings'][aj]['count_{}'.format(measure)] = len(transition_matrix[ai]['outgoings'][aj][measure])
    #print(transition_matrix)
    return transition_matrix



def save_obj(obj, name):
    import pickle
    with open('./obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    import pickle
    with open('./obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def produce_adj_mat(ts):
    all_trans = list()
    for ai in ts:
        for aj in ts[ai]['outgoings']:
            trans_name = "/".join((ai,aj))
            all_trans.append(trans_name)
    all_trans = sorted(list(set(all_trans)))

    i=0
    trans_to_int = dict()
    for trans in all_trans:
        trans_to_int[trans]=i
        i+=1

    list_of_lists = list()
    for t1 in all_trans:
        val_list = list()
        aj = t1.split('/')[1]
        for t2 in all_trans:
            ak = t2.split('/')[0]
            al = t2.split('/')[1]
            if aj==ak:
                val_list.append(1.0)
            else:
                val_list.append(0.0)
        list_of_lists.append(val_list)
    adj_mat = np.array(list_of_lists)
    return all_trans, trans_to_int, adj_mat


def visualize_ts(ts, filename='None'):
    pass
    """
    #FSM model
    fsm = FSM_Miner()
    fsm_graph = fsm._create_graph(ts, label='avg_queue_len', edge_threshold=0, start_end=False, colormap='OrRd', dashed=True)
    fsm.get_graph_info(fsm_graph)
    svg_filename = filename
    dot = fsm.get_dot(fsm_graph, svg_filename)


    #Visualizer
    app = QtWidgets.QApplication(sys.argv)

    window = Visualization()
    window.show()

    #for path in app.arguments()[1:]:
    window.load('./result/state.dot.svg');

    sys.exit(app.exec_())
    """

def save_model(model, filename):
    model_json = model.to_json()
    #with open("BPI_2012_model.json", "w") as json_file:
    with open("./models/{}.json".format(filename), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./models/{}.h5".format(filename))
    print("Saved model to disk")

def load_model(filename, loss, opt):
    from keras.layers import LSTM
    from keras.models import Sequential, Model
    from keras.layers import Dense, Input
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Nadam
    from keras.models import model_from_json
    json_file = open("./models/{}.json".format(filename), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("./models/{}.h5".format(filename))
    print("Loaded model from disk")
    #opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    loaded_model.compile(loss=loss, optimizer=opt)
    return loaded_model

def produce_2d_result(pred, int_to_trans, filename):
    import pandas as pd
    from math import pi
    from bokeh.io import show
    from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
    from bokeh.plotting import figure, output_file
    trans_list = list()
    for i in range(len(int_to_trans)):
        trans_list.append(int_to_trans[i])
    #print(trans_list)

    pred_list = list()
    label_list = list()
    time_list = list()
    count=0
    for i in range(pred.shape[0]):
        count+=1
        pred_list += list(pred[i,:])
        label_list += trans_list
        time_list += [str(count)]*pred.shape[1]

    records = list()
    records.append(('Transition', label_list))
    records.append(('Time' ,time_list))
    records.append(('Value', pred_list))

    labels = ['Transition', 'Time', 'Value']
    df = pd.DataFrame.from_items(records)
    import math
    def modi(x):
        return round(x, 2)
    df['Value'] = df['Value'].apply(modi)

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.Value.min(), high=df.Value.max())

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(x_range=[str(x) for x in list(set(df['Time'].astype(int)))], y_range=sorted(list(set(df['Transition']))),
               x_axis_location="above", plot_width=900, plot_height=400,
               tools=TOOLS, toolbar_location='below',
               tooltips=[('Transition', '@Transition'), ('Time', '@Time'), ('Value', '@Value')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="Time", y="Transition", width=1, height=1,
           source=df,
           fill_color={'field': 'Value', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    output_file("{}.html".format(filename))
    show(p)      # show the plot

def mean_absolute_percentage_error(actual, pred):
    mape_list = list()
    for i, t in enumerate(actual):
        for j, s in enumerate(t):
            if s == 0:
                continue
            else:
                ape = abs((s-pred[i][j])/(s+pred[i][j]))
            mape_list.append(ape)

    return round(np.mean(mape_list),3)

def average_correlation(actual, pred):
    import scipy
    cor_list = list()
    #actual = actual*60
    #pred = pred*60
    for i, t in enumerate(actual):
        cor = scipy.stats.pearsonr(t, pred[i])
        cor_list.append(cor)

    return round(np.mean(cor_list),3)



def test_stat_alg(folder, model, prediction, training_exp_id, _test_input, _test_output):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import pickle
    from math import sqrt
    y_pred = list()
    for i in range(_test_input.shape[2]):
        sub_test_input = _test_input[:,:,i]
        sub_test_output = _test_output[:,:,i]
        model_name = '{}_{}_{}_{}'.format(model, prediction, training_exp_id, i)
        loaded_model = pickle.load(open('./models/{}/{}.sav'.format(folder, model_name), 'rb'))
        sub_y_pred = loaded_model.predict(sub_test_input)
        y_pred.append(sub_y_pred)
    """
    flat_test_output = _test_output.flatten()
    mean = flat_test_output[np.nonzero(flat_test_output)].mean()
    std = flat_test_output[np.nonzero(flat_test_output)].std()
    median = np.median(flat_test_output[np.nonzero(flat_test_output)])
    print(mean, median, std)
    """

    y_pred = np.array(y_pred)
    y_pred = np.swapaxes(y_pred,0,1)
    if 'LR' in model:
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
    _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    mape = mean_absolute_percentage_error(_test_output, y_pred)
    corr = average_correlation(_test_output, y_pred)
    print("CORR: ", corr)
    mae = round(mean_absolute_error(_test_output, y_pred), 4)
    rmse = round(sqrt(mean_squared_error(_test_output, y_pred)), 4)
    print("Metrics of {}: {}, {}, {}".format(model, mape, mae, rmse))
    return mape, mae, rmse, mean, std, median


def test_deep_learning(model, prediction, training_exp_id, _test_input, _test_output):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from math import sqrt
    model_name = '{}_{}_{}'.format(model, prediction, training_exp_id)

    if 'SAE' in model:
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    if 'CNN' in model:
        _test_input = _test_input.reshape(_test_input.shape[0], _test_input.shape[1], _test_input.shape[2], 1)
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    elif 'LSTM' in model:
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    elif 'LRCN' in model:
        _test_input = _test_input.reshape(_test_input.shape[0], _test_input.shape[1], _test_input.shape[2], _test_input.shape[3], 1)
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])

    loss, opt = 'mae', 'adam'
    loaded_model = load_model(model_name,loss,opt)
    #score = loaded_model.evaluate(_test_input, _test_output, verbose=0)
    y_pred = loaded_model.predict(_test_input,verbose=0)
    mape = mean_absolute_percentage_error(_test_output, y_pred)
    corr = average_correlation(_test_output, y_pred)
    print("CORR: ", corr)
    mae = round(mean_absolute_error(_test_output, y_pred), 4)
    rmse = round(sqrt(mean_squared_error(_test_output, y_pred)), 4)
    #mean = np.mean(_test_output.flatten())
    flat_test_output = _test_output.flatten()
    mean = flat_test_output[np.nonzero(flat_test_output)].mean()
    std = flat_test_output[np.nonzero(flat_test_output)].std()
    median = np.median(flat_test_output[np.nonzero(flat_test_output)])
    print(mean, median, std)
    #score = loaded_model.evaluate([_test_trans_input, _test_state_input], [_test_trans_output, _test_state_output])
    print("Metrics of {}: {}, {}, {}".format(model, mape, mae, rmse))
    return mape, mae, rmse, mean, std, median

def test_hybrid_deep_learning(model, prediction, training_exp_id, _test_trans_input, _test_trans_output, _test_state_input, _test_state_output):
    if model == 'advanced_LRCN':
        _test_trans_input = _test_trans_input.reshape(_test_trans_input.shape[0], _test_trans_input.shape[1], _test_trans_input.shape[2], _test_trans_input.shape[3], 1)
        _test_trans_output = _test_trans_output.reshape(_test_trans_output.shape[0], _test_trans_output.shape[2])
        _test_state_input = _test_state_input.reshape(_test_state_input.shape[0], _test_state_input.shape[1], _test_state_input.shape[2], 1)
        _test_state_output = _test_state_output.reshape(_test_state_output.shape[0], _test_state_output.shape[2])

    loss, opt = 'mae', 'adam'
    loaded_model = load_model(model_name,loss,opt)
    score = loaded_model.evaluate([_test_trans_input, _test_state_input], [_test_trans_output, _test_state_output])
    print("Metrics of {}: {}".format(model, score))

def identify_with_deep_learning(model, prediction, training_exp_id, _test_input, _test_output):
    model_name = '{}_{}_{}'.format(model, prediction, training_exp_id)

    if 'SAE' in model:
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    if 'CNN' in model:
        _test_input = _test_input.reshape(_test_input.shape[0], _test_input.shape[1], _test_input.shape[2], 1)
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    elif 'LSTM' in model:
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])
    elif 'LRCN' in model:
        _test_input = _test_input.reshape(_test_input.shape[0], _test_input.shape[1], _test_input.shape[2], _test_input.shape[3], 1)
        _test_output = _test_output.reshape(_test_output.shape[0], _test_output.shape[2])

    loss, opt = 'mae', 'adam'
    loaded_model = load_model(model_name,loss,opt)
    pred = loaded_model.predict(_test_input)
    actual = _test_output

    act_bottleneck_list = list()
    pred_bottleneck_list = list()
    for act_perf, pred_perf in zip(actual, pred):
        #print(act_perf.shape)
        cur_act_bottleneck = act_perf.argsort()[-5:]
        cur_pred_bottleneck = pred_perf.argsort()[-5:]
        act_bottleneck_list.append(cur_act_bottleneck)
        pred_bottleneck_list.append(cur_pred_bottleneck)

    acc_list = list()
    for i, (act_bottleneck, pred_bottleneck) in enumerate(zip(act_bottleneck_list, pred_bottleneck_list)):
        if i == 0:
            continue

        new_bottleneck = [x for x in act_bottleneck if x not in act_bottleneck_list[i-1]]
        #new_bottleneck = act_bottleneck

        num_new = len(new_bottleneck)

        correct_bottleneck = [y for y in new_bottleneck if y in pred_bottleneck]

        num_correct = len(correct_bottleneck)
        if num_new !=0:
            acc_list.append(num_correct/num_new)
        else:
            acc_list.append(1)

    print(acc_list)
    from statistics import mean
    print(mean(acc_list))

def symmetric_mean_absolute_percentage_error(A, F):
    if all(A==0) and all(F==0):
        return 0
    else:
        return 100/len(A) * np.sum(np.abs(F - A) / (A+F))







