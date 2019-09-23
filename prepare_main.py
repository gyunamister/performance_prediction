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
	#python prepare_main.py --exp_name 'BPIC19' --path './sample_data/BPIC2019.csv' --agg 'avg' --measure 'sojourn' --fv_format '2d' --start "2018-03-01 00:00:00" --end "2018-09-30 23:59:59" --range '0,0,0,72,0,0' --stride '0,0,0,12,0,0' --input_size 10 --output_size 1 --edge_threshold 500 --node_threshold 1000
	#python prepare_main.py --exp_name 'BPIC12' --path './sample_data/BPIC12.csv' --agg 'avg' --measure 'processing' --fv_format '2d' --start "2011-10-01 00:00:00" --end "2012-03-14 23:59:59" --range '0,0,0,1,0,0' --stride '0,0,0,1,0,0' --input_size 24 --output_size 1 --edge_threshold 100
	exp_name = 'D0909'
	path = './sample_data/helpdesk.csv'
	agg = 'avg'
	measure = 'sojourn'
	fv_format = '2d'
	#s_month, s_day, s_hour, s_min = '01', '01', '00', '00'
	#e_month, e_day, e_hour, e_min = '07', '31', '23', '59'
	#r_day, r_hour, r_min = 0, 1, 0
	#stride_day, stride_hour, stride_min = 0, 1, 0
	start = "2018-01-15 00:00:00"
	end = "2018-10-30 23:59:59"
	input_size = 15*24
	output_size = 5*24
	#edge_threshold = 500

	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str)
	parser.add_argument('--path', type=str)
	parser.add_argument('--agg', type=str, choices=['avg', 'length', 'count', 'std'])
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

	FLAGS = parser.parse_args()

	exp_name = FLAGS.exp_name
	path = FLAGS.path
	agg = FLAGS.agg
	measure = FLAGS.measure
	fv_format = FLAGS.fv_format

	start = FLAGS.start
	s_date, s_time = [x for x in start.split(" ")]
	"""
	s_date, s_time = [x for x in start.split(" ")]
	s_year, s_month, s_day = [x for x in s_date.split("-")]
	s_hour, s_min, s_sec = [x for x in s_time.split(":")]
	"""

	end = FLAGS.end
	e_date, e_time = [x for x in start.split(" ")]
	"""
	e_date, e_time = [x for x in start.split(" ")]
	e_year, e_month, e_day = [x for x in e_date.split("-")]
	e_hour, e_min, e_sec = [x for x in e_time.split(":")]
	"""

	tw_range = FLAGS.range
	r_year, r_month, r_day, r_hour, r_min, r_sec = [int(x) for x in tw_range.split(",")]
	"""
	r_date, r_time = [x for x in tw_range.split(" ")]
	r_year, r_month, r_day = [int(x) for x in r_date.split("-")]
	r_hour, r_min, r_sec = [int(x) for x in r_time.split(":")]
	"""

	tw_stride = FLAGS.stride
	stride_year, stride_month, stride_day, stride_hour, stride_min, stride_sec = [int(x) for x in tw_stride.split(",")]
	"""
	stride_date, stride_time = [x for x in tw_range.split(" ")]
	stride_year, stride_month, stride_day = [int(x) for x in stride_date.split("-")]
	stride_hour, stride_min, stride_sec = [int(x) for x in stride_time.split(":")]
	"""

	input_size = FLAGS.input_size
	output_size = FLAGS.output_size
	edge_threshold = FLAGS.edge_threshold
	node_threshold = FLAGS.node_threshold

	exp_id = '{}_{}_{}_{}_{}-{}_range{}_stride{}_input{}_output{}_{}_{}'.format(exp_name, agg, measure, fv_format, s_date, e_date, tw_range, tw_stride, input_size, output_size, edge_threshold, node_threshold)

	#eventlog import
	#path = './sample_data/0314_Hospital_ED.csv'
	case, activity, timestamp = 'CASEOID', 'ACTIVITYOID', 'TIMESTAMP'

	pp = Preprocessor()
	eventlog = pp.load_eventlog(path, case, activity, timestamp, clear=False)
	if measure == 'processing' or measure == 'waiting':
		eventlog.assign_timestamp('START', new_name='START_TIMESTAMP')



	#Reference TS
	TM = TransitionMatrix()
	transition_matrix = TM.get_transition_matrix(eventlog, workers=4, abs_type='sequence', horizon=1)
	transition_matrix = apply_node_threshold(transition_matrix, node_threshold)
	transition_matrix = apply_edge_threshold(transition_matrix, edge_threshold)

	fg = FeatureGenerator()
	#save mapping dict
	trans_to_int, int_to_trans, states_to_int, int_to_states = fg.produce_mapping_dict(transition_matrix, exp_id)

	#window list
	range_days = r_day*0
	range_seconds = r_hour*60*60 + r_min*60
	stride_days = stride_day*0
	stride_seconds = stride_hour*60*60 + stride_min*60
	window_list = fg.produce_window(start=start, end=end, range_days=range_days, range_seconds=range_seconds, stride_days=stride_days, stride_seconds=stride_seconds)
	#feature vector 및 input&output 생성
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
		"""
		from sklearn.model_selection import train_test_split
		trans_train_X, trans_test_X, trans_train_y, trans_test_y = train_test_split(trans_X, trans_y, test_size=0.2, random_state=42)
		state_train_X, state_test_X, state_train_y, state_test_y = train_test_split(state_X, state_y, test_size=0.2, random_state=42)
		print("trans_train: {}, {}".format(trans_train_X.shape, trans_train_y.shape))
		print("trans_test: {}, {}".format(trans_test_X.shape, trans_test_y.shape))
		print("state_train: {}, {}".format(state_train_X.shape, state_train_y.shape))
		print("state_test: {}, {}".format(state_test_X.shape, state_test_y.shape))
		"""

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

		"""
		from sklearn.model_selection import train_test_split
		trans_train_X, trans_test_X, trans_train_y, trans_test_y = train_test_split(trans_X, trans_y2, test_size=0.2, random_state=42)
		state_train_X, state_test_X, state_train_y, state_test_y = train_test_split(state_X, state_y2, test_size=0.2, random_state=42)
		print("trans_train: {}, {}".format(trans_train_X.shape, trans_train_y.shape))
		print("trans_test: {}, {}".format(trans_test_X.shape, trans_test_y.shape))
		print("state_train: {}, {}".format(state_train_X.shape, state_train_y.shape))
		print("state_test: {}, {}".format(state_test_X.shape, state_test_y.shape))
		"""

	#save npy files
	if fv_format == '2d':
		np.save('./matrices/trans_train_X_{}.npy'.format(exp_id), trans_X)
		np.save('./matrices/trans_train_y_{}.npy'.format(exp_id), trans_y)
		#np.save('./matrices/trans_test_X_{}.npy'.format(exp_id), trans_test_X)
		#np.save('./matrices/trans_test_y_{}.npy'.format(exp_id), trans_test_y)

		np.save('./matrices/state_train_X_{}.npy'.format(exp_id), state_X)
		np.save('./matrices/state_train_y_{}.npy'.format(exp_id), state_y)
		#np.save('./matrices/state_test_X_{}.npy'.format(exp_id), state_test_X)
		#np.save('./matrices/state_test_y_{}.npy'.format(exp_id), state_test_y)
	else:
		np.save('./matrices/trans_train_X_{}.npy'.format(exp_id), trans_X)
		np.save('./matrices/trans_train_y_{}.npy'.format(exp_id), trans_y2)
		#np.save('./matrices/trans_test_X_{}.npy'.format(exp_id), trans_test_X)
		#np.save('./matrices/trans_test_y_{}.npy'.format(exp_id), trans_test_y)

		np.save('./matrices/state_train_X_{}.npy'.format(exp_id), state_X)
		np.save('./matrices/state_train_y_{}.npy'.format(exp_id), state_y2)
		#np.save('./matrices/state_test_X_{}.npy'.format(exp_id), state_test_X)
		#np.save('./matrices/state_test_y_{}.npy'.format(exp_id), state_test_y)