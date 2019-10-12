#-*- coding: utf-8 -*-
import sys
import os
import signal
import copy
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


p = Path(__file__).resolve().parents[1]
sys.path.append(os.path.abspath(str(p)))

from PyProM.src.mining.transition_matrix import TransitionMatrix

from preprocess.preprocess import Preprocessor
from preprocess.feature_generator import FeatureGenerator
from preprocess.util import *



if __name__ == '__main__':
	#parameter setting
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str)
	parser.add_argument('--path', type=str)
	parser.add_argument('--agg', type=str, choices=['avg'])
	parser.add_argument('--measure', type=str, choices=['waiting', 'processing', 'sojourn'])
	parser.add_argument('--fv_format', type=str, choices=['2d', '3d'])
	parser.add_argument('--start', type=str)
	parser.add_argument('--end', type=str)
	parser.add_argument('--range', type=str)
	parser.add_argument('--stride', type=str)
	parser.add_argument('--input_size', type=int)
	parser.add_argument('--output_size', type=int)
	parser.add_argument('--edge_threshold', type=int)
	parser.add_argument('--node_threshold', type=int)
	parser.add_argument('--horizon', type=int)

	FLAGS = parser.parse_args()

	exp_name = FLAGS.exp_name
	path = FLAGS.path
	agg = FLAGS.agg
	measure = FLAGS.measure
	fv_format = FLAGS.fv_format

	start = FLAGS.start
	s_date, s_time = [x for x in start.split(" ")]
	end = FLAGS.end
	e_date, e_time = [x for x in start.split(" ")]

	tw_range = FLAGS.range
	r_year, r_month, r_day, r_hour, r_min, r_sec = [int(x) for x in tw_range.split(",")]

	tw_stride = FLAGS.stride
	stride_year, stride_month, stride_day, stride_hour, stride_min, stride_sec = [int(x) for x in tw_stride.split(",")]

	input_size = FLAGS.input_size
	output_size = FLAGS.output_size

	edge_threshold = FLAGS.edge_threshold
	node_threshold = FLAGS.node_threshold
	horizon = FLAGS.horizon

	exp_id = '{}_{}_{}_{}_{}-{}_range{}_stride{}_input{}_output{}_{}_{}_{}'.format(exp_name, agg, measure, fv_format, s_date, e_date, tw_range, tw_stride, input_size, output_size, edge_threshold, node_threshold, horizon)

	#eventlog import
	case, activity, timestamp = 'CASEOID', 'ACTIVITYOID', 'TIMESTAMP'
	pp = Preprocessor()
	eventlog = pp.load_eventlog(path, case, activity, timestamp, clear=False)
	if measure == 'processing' or measure == 'waiting':
		eventlog.assign_timestamp('START', new_name='START_TIMESTAMP')

	#produce reference TS
	TM = TransitionMatrix()
	transition_matrix = TM.get_transition_matrix(eventlog, workers=4, abs_type='sequence', horizon=horizon)
	transition_matrix = apply_node_threshold(transition_matrix, node_threshold)
	transition_matrix = apply_edge_threshold(transition_matrix, edge_threshold)

	#produce features
	fg = FeatureGenerator()
	#save mapping dict
	trans_to_int, int_to_trans, states_to_int, int_to_states = fg.produce_mapping_dict(transition_matrix, exp_id)

	#produce time winodws
	range_days = r_day*0
	range_seconds = r_hour*60*60 + r_min*60
	stride_days = stride_day*0
	stride_seconds = stride_hour*60*60 + stride_min*60
	window_list = fg.produce_window(start=start, end=end, range_days=range_days, range_seconds=range_seconds, stride_days=stride_days, stride_seconds=stride_seconds)

	#produce feature vector and training sets
	#produce 2-dim process representation matrix
	if fv_format=='2d':
		if measure == 'processing' or measure == 'waiting':
			ts_list = fg.replay_log(TM, window_list, eventlog, transition_matrix, start_time='START_TIMESTAMP', complete_time='TIMESTAMP', measure=measure, agg=agg)
		else:
			ts_list = fg.replay_log(TM, window_list, eventlog, transition_matrix, start_time='default', complete_time='TIMESTAMP', measure=measure, agg=agg)

		perf_measure = '{}_{}'.format(agg, measure)
		fv_trans = fg.produce_2d_feature_vector(ts_list, perf_measure, trans_to_int, int_to_trans)
		fv_state = fg.produce_2d_state_feature_vector(ts_list, perf_measure, states_to_int, int_to_states)

		#save feature vector for search-based method
		np.save('./matrices/fv_trans_{}.npy'.format(exp_id), fv_trans)
		np.save('./matrices/fv_state_{}.npy'.format(exp_id), fv_state)

		#3d input generation (samples, num_trans, num_window)
		trans_X, trans_y = fg.produce_3d_samples(fv_trans, input_size=input_size, output_size=output_size)
		state_X, state_y = fg.produce_3d_samples(fv_state, input_size=input_size, output_size=output_size)

		print("trans: {}, {}".format(trans_X.shape, trans_y.shape))
		print("state: {}, {}".format(state_X.shape, state_y.shape))
	#produce 3-dim process representation matrix
	elif fv_format=='3d':
		if measure == 'processing' or measure == 'waiting':
			ts_list = fg.replay_log(TM, window_list, eventlog, transition_matrix, start_time='START_TIMESTAMP', complete_time='TIMESTAMP', measure=measure, agg=agg)
		else:
			ts_list = fg.replay_log(TM, window_list, eventlog, transition_matrix, start_time='default', complete_time='TIMESTAMP', measure=measure, agg=agg)

		perf_measure = '{}_{}'.format(agg, measure)
		fv_trans = fg.produce_3d_feature_vector(ts_list, perf_measure, states_to_int, int_to_states)
		fv2_trans = fg.produce_2d_feature_vector(ts_list, perf_measure, trans_to_int, int_to_trans)
		fv_state = fg.produce_3d_feature_vector(ts_list, perf_measure, states_to_int, int_to_states)
		fv2_state = fg.produce_2d_state_feature_vector(ts_list, perf_measure, states_to_int, int_to_states)

		#4d input generation (samples, num_nodes, num_nodes, num_window)
		trans_X, trans_y = fg.produce_4d_samples(fv_trans, input_size=input_size, output_size=output_size)
		trans_X2, trans_y2 = fg.produce_3d_samples(fv2_trans, input_size=input_size, output_size=output_size)
		state_X, state_y = fg.produce_4d_samples(fv_state, input_size=input_size, output_size=output_size)
		state_X2, state_y2 = fg.produce_3d_samples(fv2_state, input_size=input_size, output_size=output_size)

		print("trans: {}, {}".format(trans_X.shape, trans_y.shape))
		print("state: {}, {}".format(state_X.shape, state_y.shape))

	#save npy files
	if fv_format == '2d':
		np.save('./matrices/trans_train_X_{}.npy'.format(exp_id), trans_X)
		np.save('./matrices/trans_train_y_{}.npy'.format(exp_id), trans_y)

		np.save('./matrices/state_train_X_{}.npy'.format(exp_id), state_X)
		np.save('./matrices/state_train_y_{}.npy'.format(exp_id), state_y)
	else:
		np.save('./matrices/trans_train_X_{}.npy'.format(exp_id), trans_X)
		np.save('./matrices/trans_train_y_{}.npy'.format(exp_id), trans_y2)

		np.save('./matrices/state_train_X_{}.npy'.format(exp_id), state_X)
		np.save('./matrices/state_train_y_{}.npy'.format(exp_id), state_y2)