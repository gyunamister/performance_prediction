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

from preprocess.util import *
from prediction.models import *

from sklearn import preprocessing

if __name__ == '__main__':
	#python train_main.py --fv_format '2d' --prediction 'state' --exp_id 'BPIC15_avg_sojourn_2d_2011-02-01-2011-02-01_range0,0,0,72,0,0_stride0,0,0,24,0,0_input30_output1_500' --algo 'basic_SVR'
	#python train_main.py --fv_format '2d' --prediction 'state' --exp_id 'helpdesk_avg_sojourn_2d_2010-02-01-2010-02-01_range0,0,0,72,0,0_stride0,0,0,24,0,0_input15_output1_0' --algo 'basic_CNN'
	parser = argparse.ArgumentParser()
	parser.add_argument('--search', type=str, choices=['True', 'False'])
	parser.add_argument('--fv_format', type=str, choices=['2d', '3d'])
	parser.add_argument('--prediction', type=str, choices=['state', 'trans'])
	parser.add_argument('--exp_id', type=str)
	parser.add_argument('--algo', type=str)
	parser.add_argument('--result', type=str)
	FLAGS, unparsed = parser.parse_known_args()

	fv_format = FLAGS.fv_format
	prediction = FLAGS.prediction
	exp_id = FLAGS.exp_id
	algo = FLAGS.algo
	result = FLAGS.result
	agg = exp_id.split('_')[1]

	search_based = False
	if FLAGS.search == 'True':
		search_based = True
	else:
		search_based = False

	if search_based == True:
		fv = np.load('./matrices/fv_{}_{}.npy'.format(prediction, exp_id))
		from sklearn.model_selection import KFold
		kfold = KFold(n_splits=5, shuffle=True, random_state=1)
		euc_list = list()
		che_list = list()
		cos_list = list()
		metrics_list = [euc_list, che_list, cos_list]
		for j, (train, validation) in enumerate(list(kfold.split(fv))):
			training_fv = fv[train]
			test_fv = fv[validation]
			if agg == 'avg' or agg == 'std':
				training_fv, test_fv = training_fv/60, test_fv/60
			from scipy.spatial.distance import euclidean, chebyshev, cosine
			dist_func_list = [euclidean, chebyshev, cosine]
			for i, dist_func in enumerate(dist_func_list):
				mae, mape = search_based_method(training_fv, test_fv, dist_func)
				metrics_list[i].append([mae,mape])
		with open("./result/{}.txt".format(result), "a") as f:
			f.write("{} - exp_id: {} \n EUC: \n accuracy: {} \n".format(prediction, exp_id, metrics_list[0]))
			f.write("{} - exp_id: {} \n CHE: \n accuracy: {} \n".format(prediction, exp_id, metrics_list[1]))
			f.write("{} - exp_id: {} \n COS: \n accuracy: {} \n".format(prediction, exp_id, metrics_list[2]))
		import sys
		sys.exit("Search Done")

	#load npy files
	if fv_format=='2d':
		_input = np.load('./matrices/{}_train_X_{}.npy'.format(prediction, exp_id))

		_output = np.load('./matrices/{}_train_y_{}.npy'.format(prediction, exp_id))
		if agg == 'avg':
			_input, _output = _input/(60), _output/(60)
	elif fv_format=='3d':
		if prediction=='both':
			_trans_input = np.load('./matrices/trans_train_X_{}.npy'.format(exp_id))
			_trans_output = np.load('./matrices/trans_train_y_{}.npy'.format(exp_id))
			_state_input = np.load('./matrices/state_train_X_{}.npy'.format(exp_id))
			_state_output = np.load('./matrices/state_train_y{}.npy'.format(exp_id))
			if agg == 'avg':
				_trans_input, _trans_output, _state_input, _state_output = _trans_input/(60), _trans_output/(60), _state_input/(60), _state_output/(60)
		else:
			_input = np.load('./matrices/{}_train_X_{}.npy'.format(prediction, exp_id))

			_output = np.load('./matrices/{}_train_y_{}.npy'.format(prediction, exp_id))
			if agg == 'avg':
				_input, _output = _input/60, _output/60

	#LSTM
	if algo == 'basic_LSTM':
		model_name = 'basic_LSTM_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_LSTM, _input, _output)



	#CNN
	elif algo == 'basic_CNN':
		model_name = 'basic_CNN_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_CNN, _input, _output)



	#basic_LRCN
	elif algo == 'basic_LRCN':
		#model = basic_LRCN(_input, _output)
		model_name = 'basic_LRCN_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_LRCN, _input, _output)
		#save_model(model, model_name)

	#Linear Regression
	elif algo == 'basic_LR':
		model_name = 'basic_LR_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_linear_regression, _input, _output)

	#Random Forest
	elif algo == 'basic_RF':
		model_name = 'basic_RF_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_random_forest, _input, _output)

	#SVR
	elif algo == 'basic_SVR':
		model_name = 'basic_SVR_{}_{}'.format(prediction, exp_id)
		accuracy = train_and_evaluate(model_name, basic_SVR, _input, _output)

	else:
		raise("algorithm should be given")
	print("{}: {} training completed".format(algo,exp_id))


	with open("./result/{}.txt".format(result), "a") as f:
		f.write("{} - exp_id: {} \n algo: {} \n accuracy: {} \n".format(prediction, exp_id, algo, accuracy))