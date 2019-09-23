from preprocess.util import *

class FeatureGenerator(object):
    def produce_mapping_dict(self, ts, exp_id):
        trans_to_int = dict()
        int_to_trans = dict()
        states_to_int = dict()
        int_to_states = dict()
        i = 0
        j = 0
        for ai in ts:
            states_to_int[ai] = i
            int_to_states[i] = ai
            i+=1
            for aj in ts[ai]['outgoings']:
                trans_name = "/".join((ai,aj))
                trans_to_int[trans_name] = j
                int_to_trans[j] = trans_name
                j+=1
        save_obj(trans_to_int, "trans_to_int_{}".format(exp_id))
        save_obj(int_to_trans, "int_to_trans_{}".format(exp_id))
        save_obj(states_to_int, "states_to_int_{}".format(exp_id))
        save_obj(int_to_states, "int_to_states_{}".format(exp_id))

        return trans_to_int, int_to_trans, states_to_int, int_to_states

    def produce_states_list(self, ts):
        all_nodes = list(set([ai for ai in ts]))
        all_nodes.remove('START')
        all_nodes = sorted(all_nodes)
        return all_nodes

    def produce_window(self, start, end, range_days=0, range_seconds=0, stride_days=0, stride_seconds=0):
        import datetime
        window_list = list()
        range_timedelta = datetime.timedelta(range_days, range_seconds)
        stride_timedelta = datetime.timedelta(stride_days, stride_seconds)
        start, end = get_dt_from_string(start), get_dt_from_string(end)
        dt1, dt2 = start, start
        while dt2 < end:
            dt2 = dt1 + range_timedelta
            str_dt1 = get_string_from_dt(dt1)
            str_dt2 = get_string_from_dt(dt2)
            window_list.append((str_dt1, str_dt2))
            dt1 = dt1 + stride_timedelta
        return window_list

    def replay_log(self, TM, window_list, eventlog, transition_matrix, start_time, complete_time, measure='processing', agg='avg'):
        replay_list = list()
        for dt1, dt2 in window_list:
            #only complete timestamp
            if start_time == 'default':
                filtered_log = filter_log_by_timestamp(eventlog, start_time=start_time, complete_time=complete_time, dt1=dt1, dt2=dt2)
            else:
                filtered_log = filter_log_by_timestamp_with_start(eventlog, start_time=start_time, complete_time=complete_time, dt1=dt1, dt2=dt2, measure=measure, agg=agg)
            perf_measure = '{}_{}'.format(agg, measure)
            annotated_ts = TM.clear_annotation(transition_matrix, perf_measure)

            #measurements
            if measure == 'processing':
                annotated_ts = TM.annotate_transition_matrix(filtered_log, 4, transition_matrix, start_time=start_time, complete_time=complete_time, value='processing')
            elif measure == 'waiting':
                annotated_ts = TM.annotate_transition_matrix(filtered_log, 4, transition_matrix, start_time=start_time, complete_time=complete_time, value='waiting')
            elif measure == 'sojourn':
                annotated_ts = TM.annotate_transition_matrix(filtered_log, 4, transition_matrix, start_time=start_time, complete_time=complete_time, value='sojourn')

            #aggregation
            if agg == 'avg':
                annotated_ts = compute_avg_time(annotated_ts, measure)
            elif agg == 'length':
                timestamp_vals = filtered_log.get_col_values('TIMESTAMP')
                log_start_at = min(timestamp_vals)
                log_end_at = max(timestamp_vals)
                log_range = log_end_at-log_start_at
                annotated_ts = compute_avg_queue_len(annotated_ts, measure, log_range)
            elif agg == 'count':
                annotated_ts = compute_cnt(annotated_ts, measure)
            elif agg == 'std':
                annotated_ts = compute_std_time(annotated_ts, measure)
            replay_list.append(annotated_ts)
        return replay_list


    def produce_2d_feature_vector(self, ts_list, perf_measure, trans_to_int, int_to_trans):
        #(num_trans, num_window)
        feature_vector = list()
        num_acrs = 0
        i = 0
        num_acrs += len(trans_to_int)
        #print("num arcs: {}".format(num_acrs))
        for ts in ts_list:
            t_row = [0 for x in range(num_acrs)]
            for ai in ts:
                for aj in ts[ai]['outgoings']:
                    trans_name = "/".join((ai,aj))
                    if trans_name in trans_to_int:
                        val = ts[ai]['outgoings'][aj][perf_measure]
                        #print("{}->{}: {}".format(ai,aj,val))
                        idx = trans_to_int[trans_name]
                        t_row[idx] = val
                    else:
                        continue
            t_row = np.array(t_row)
            t_row = np.hstack(t_row)
            feature_vector.append(t_row)
        feature_vector=np.array(feature_vector)
        print(feature_vector)
        return feature_vector

    def produce_2d_state_feature_vector(self, ts_list, perf_measure, states_to_int, int_to_states):
        #(num_trans, num_window)

        feature_vector = list()
        num_nodes = len(states_to_int)
        for ts in ts_list:
            t_row = [0 for x in range(num_nodes)]
            for ai in ts:
                if ai in states_to_int:
                    val = ts[ai][perf_measure]
                    idx = states_to_int[ai]
                    t_row[idx] = val
                else:
                    continue
            t_row = np.array(t_row)
            t_row = np.hstack(t_row)
            feature_vector.append(t_row)
        feature_vector=np.array(feature_vector)
        return feature_vector



    def produce_3d_samples(self, fv, input_size=1, output_size=1):
        #generate (samples, num_trans, num_window)
        X_train = list()
        y_train = list()
        for i in range(0,fv.shape[0]-input_size-output_size+1,1):
            X = fv[i:i+input_size,:]
            y = fv[i+input_size:i+input_size+output_size,:]
            X_train.append(X)
            y_train.append(y)
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        return X_train, y_train

    def produce_3d_feature_vector(self, ts_list, perf_measure, states_to_int, int_to_states):
        #(num_window, num_nodes, num_nodes)
        feature_vector = list()

        all_nodes = list(states_to_int.keys())

        #print("num arcs: {}".format(num_acrs))
        list_of_trans_mat = list()
        for ts in ts_list:
            list_of_lists = list()
            for ai in all_nodes:
                if ai not in ts:
                    val_list = [0] * len(all_nodes)
                    continue
                val_list = list()
                for aj in all_nodes:
                    if aj in ts[ai]['outgoings'].keys():
                        val_list.append(ts[ai]['outgoings'][aj][perf_measure])
                    else:
                        val_list.append(0)
                list_of_lists.append(val_list)
            list_of_trans_mat.append(list_of_lists)
        array_of_trans_mat = np.array(list_of_trans_mat)
        return array_of_trans_mat


    def produce_4d_samples(self, fv, input_size=3, output_size=1):
        #generate (samples, num_nodes, num_nodes, num_window)
        X_train = list()
        y_train = list()
        for i in range(0,fv.shape[0]-input_size-output_size+1,1):
            X = fv[i:i+input_size,:,:]
            y = fv[i+input_size:i+input_size+output_size,:,:]
            X_train.append(X)
            y_train.append(y)
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        return X_train, y_train
