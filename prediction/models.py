import pickle
from sklearn.metrics import mean_absolute_error
from preprocess.util import *
def get_callbacks(model_path, patience_lr, patience_es):
	from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
	mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_es)
	return [mcp_save, reduce_lr_loss, es]

def train_and_evaluate(model_name, get_model, X, y):
	from sklearn.model_selection import KFold
	kfold = KFold(n_splits=5, shuffle=True, random_state=1)
	batch_size = 16
	droprate = 0.25
	epochs = 100
	accuracy = []

	if 'LSTM' in model_name or 'CNN' in model_name or 'LRCN' in model_name:
		dnn = True
	else:
		dnn = False

	for j, (train, validation) in enumerate(list(kfold.split(X, y))):
		model_path = "./prediction/models/{}-fold{}.h5".format(model_name,j)
		if dnn == True:
			callbacks = get_callbacks(model_path, patience_lr=3, patience_es=5)
			k_accuracy = get_model(model_name, X, y, train, validation, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
		else:
			y_pred, y_actual = list(), list()
			for i in range(y.shape[2]):
				sub_X = X[:,:,i]
				sub_y = y[:,:,i]
				sub_y_pred, sub_y_actual = get_model(model_name, model_path, sub_X, sub_y, train, validation)
				y_pred.append(sub_y_pred)
				y_actual.append(sub_y_actual)
			y_pred = np.array(y_pred)
			print(y_pred.shape)
			y_pred = np.swapaxes(np.array(y_pred),0,1)
			y_actual = np.array(y_actual)
			print(y_actual.shape)
			y_pred = np.concatenate(y_pred, axis=0)
			y_actual = np.concatenate(y_actual, axis=0)
			print(y_actual.shape)
			if 'LR' in model_name:
				y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
			else:
				y_pred = y_pred.reshape(y_pred.shape[0], 1)
			#y_actual = y_actual.reshape(y_actual.shape[0], y_actual.shape[2])
			mae = round(mean_absolute_error(y_actual, y_pred),4)
			mape = mean_absolute_percentage_error(y_actual, y_pred)
			k_accuracy = [mae, mape]
			print(k_accuracy)
		accuracy.append(k_accuracy)
	return accuracy


def basic_LSTM(model_name, X, y, train, validation, epochs, batch_size, droprate=0.25, verbose=1, callbacks=[]):
	from keras.layers import LSTM
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input
	from keras.callbacks import EarlyStopping, ModelCheckpoint
	from keras.layers.normalization import BatchNormalization
	from keras.optimizers import Nadam

	y = y.reshape(y.shape[0], y.shape[2])
	input_shape = (X.shape[1], X.shape[2])
	output_shape = y.shape[1]

	model = Sequential()
	model.add(LSTM(X.shape[2]*4, input_shape=input_shape, return_sequences=True))
	model.add(BatchNormalization())
	model.add(LSTM(X.shape[2]*2, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Dense(output_shape))
	model.compile(loss='mae', optimizer='adam', metrics=['mae'])
	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]

	#train the model
	model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(val_X, val_y), callbacks=callbacks)

	# evaluation
	y_pred = model.predict(val_X)
	mape = mean_absolute_percentage_error(val_y, y_pred)
	mae = round(mean_absolute_error(val_y, y_pred), 4)
	k_accuracy = [mae, mape]
	print(k_accuracy)
	return k_accuracy

def basic_CNN(model_name, X, y, train, validation, epochs, batch_size, droprate=0.25, verbose=1, callbacks=[]):
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Activation
	#reshape input/output
	X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
	y = y.reshape(y.shape[0], y.shape[2])
	input_shape = (X.shape[1],X.shape[2],X.shape[3])
	output_shape = y.shape[1]

	#parameters
	filter_pixel = 3

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(16, kernel_size=(filter_pixel, filter_pixel), activation='relu', input_shape=input_shape, padding="same"))
	model.add(BatchNormalization())
	model.add(Dropout(droprate))#3
	model.add(Conv2D(8, kernel_size=(filter_pixel, filter_pixel), activation='relu', border_mode="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(droprate))#3
	model.add(Flatten())
	model.add(Dense(y.shape[1]*2,use_bias=False)) #13
	model.add(Activation('relu')) #14
	model.add(BatchNormalization())
	model.add(Dropout(droprate))      #15
	model.add(Dense(output_shape))

	#compile model
	model.compile(loss='mae', optimizer='RMSprop', metrics=['mae'])

	#if normalizing input/output
	"""
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import MinMaxScaler

	input_scaler = MinMaxScaler()
	output_scaler = StandardScaler()
	#input_scaler.fit(X[train])
	#train_X = input_scaler.transform(X[train])
	x_min = X[train].min(axis=(1,2), keepdims=True)
	x_max = X[train].max(axis=(1,2), keepdims=True)
	# 모두 0인 경우 에러 발생
	train_X = (X[train] - x_min) / (x_max - x_min)
	x_min = X[validation].min(axis=(1,2), keepdims=True)
	x_max = X[validation].max(axis=(1,2), keepdims=True)

	val_X = (X[validation] - x_min) / (x_max - x_min)
	#val_X = input_scaler.transform(X[validation])
	#output_scaler.fit(y[train])
	#train_y = output_scaler.transform(y[train])
	#val_y = output_scaler.transform(y[validation])
	train_y = y[train]
	val_y = y[validation]
	"""

	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]

	#train the model
	model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(val_X, val_y), callbacks=callbacks)

	# evaluation
	y_pred = model.predict(val_X)
	#if normalizing input/output
	#y_pred = output_scaler.inverse_transform(y_pred)
	"""
	for i in range(len(y_pred)):
		print("Pred: {}, Actual: {}".format(y_pred[i], val_y[i]))
	"""
	mape = mean_absolute_percentage_error(val_y, y_pred)
	mae = round(mean_absolute_error(val_y, y_pred), 4)
	k_accuracy = [mae, mape]
	print(k_accuracy)
	return k_accuracy

def basic_LRCN(model_name, X, y, train, validation, epochs, batch_size, droprate=0.25, verbose=1, callbacks=[]):
	from keras.models import Sequential
	from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, GlobalAveragePooling1D, LSTM, BatchNormalization
	from keras.optimizers import Adam
	from keras.callbacks import EarlyStopping
	from keras.utils import plot_model, np_utils

	X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
	y = y.reshape(y.shape[0], y.shape[2])
	input_shape = (X.shape[1],X.shape[2],X.shape[3], 1)
	output_shape = y.shape[1]

	#parameters
	filter_pixel = 3
	frames = 5 # The number of frames for each sequence

	#create model
	model=Sequential()
	#add model layers
	model.add(TimeDistributed(Conv2D(16, (filter_pixel, filter_pixel), padding='same'), input_shape=input_shape))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(BatchNormalization()))
	#model.add(TimeDistributed(Dropout(droprate)))#3
	model.add(TimeDistributed(Conv2D(8, (filter_pixel, filter_pixel))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Dropout(droprate)))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(output_shape*2)))
	model.add(TimeDistributed(Dropout(droprate)))
	model.add(LSTM(output_shape*4, name="lstm_layer", return_sequences=True));
	model.add(LSTM(output_shape*4, name="lstm_layer_2"));
	model.add(Dense(output_shape, name="last_dense"))

	#compile model
	model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
	#plot_model(model, to_file='../models/cnn_lstm.png')
	#model.fit(X_train, y_train, epochs=epochs, validation_split=0.0, batch_size=batch_size, verbose=1)

	#train the model
	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]
	model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(val_X, val_y), callbacks=callbacks)

	# evaluation
	y_pred = model.predict(val_X)
	mape = mean_absolute_percentage_error(val_y, y_pred)
	mae = round(mean_absolute_error(val_y, y_pred), 4)
	k_accuracy = [mae, mape]
	print(k_accuracy)
	return k_accuracy

def basic_linear_regression(model_name, model_path, X, y, train, validation):
	from sklearn import linear_model
	model = linear_model.LinearRegression()

	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]

	# Train the model using the training sets
	model.fit(train_X, train_y)

	# evaluation
	sub_y_pred = model.predict(val_X)

	# save the model to disk
	#pickle.dump(model, open(model_path, 'wb'))

	return sub_y_pred, val_y

def basic_random_forest(model_name, model_path, X, y, train, validation):
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]

	model.fit(train_X, train_y)

	# evaluation
	sub_y_pred = model.predict(val_X)

	# save the model to disk
	pickle.dump(model, open(model_path, 'wb'))

	return sub_y_pred, val_y

def basic_SVR(model_name, model_path, X, y, train, validation):
	#print(X_train.shape, y_train.shape)
	from sklearn.svm import SVR
	model = SVR(gamma='scale', C=1.0, epsilon=0.2)

	train_X, train_y, val_X, val_y = X[train], y[train], X[validation], y[validation]

	model.fit(train_X, train_y)

	# evaluation
	sub_y_pred = model.predict(val_X)

	# save the model to disk
	pickle.dump(model, open(model_path, 'wb'))

	return sub_y_pred, val_y

def advanced_LRCN(trans_X_train, trans_y_train, state_X_train, state_y_train):
	from keras.layers import Input, Dense, TimeDistributed, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, LSTM, BatchNormalization, concatenate
	from keras.optimizers import Adam
	from keras.models import Model
	batch_size = 16
	epochs = 10
	trans_X_train = trans_X_train.reshape(trans_X_train.shape[0], trans_X_train.shape[1], trans_X_train.shape[2], trans_X_train.shape[3], 1)
	trans_y_train = trans_y_train.reshape(trans_y_train.shape[0], trans_y_train.shape[2])
	state_X_train = state_X_train.reshape(state_X_train.shape[0], state_X_train.shape[1], state_X_train.shape[2], 1)
	state_y_train = state_y_train.reshape(state_y_train.shape[0], state_y_train.shape[2])
	def first_model(X_train, y_train):
		frames = 5 # The number of frames for each sequence
		input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3], 1)

		first_input = Input(shape=input_shape, name='first_input')

		droprate = 0.25
		filter_pixel = 3


		#model=Sequential()
		conv1 = TimeDistributed(Conv2D(16, (filter_pixel, filter_pixel), padding='same'))(first_input)
		act1 = TimeDistributed(Activation('relu'))(conv1)
		norm1 = TimeDistributed(BatchNormalization())(act1)
		conv2 = TimeDistributed(Conv2D(8, (filter_pixel, filter_pixel)))(norm1)
		act2 = TimeDistributed(Activation('relu'))(conv2)
		pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(act2)
		dropout1 = TimeDistributed(Dropout(droprate))(pool1)

		flatten1 = TimeDistributed(Flatten())(dropout1)
		dense1 = TimeDistributed(Dense(y_train.shape[1]*2))(flatten1)
		lstm1 = LSTM(y_train.shape[1]*4, name="lstm_layer", return_sequences=True)(dense1)
		lstm2 = LSTM(y_train.shape[1]*2, name="lstm_layer_2")(lstm1)
		return first_input, lstm2

	def second_model(X_train, y_train):
		input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
		#size of parameters
		batch_size = 16
		droprate=0.25
		filter_pixel = 3

		#create model
		second_input = Input(shape=input_shape, name='second_input')
		#add model layers
		conv1 = Conv2D(32, kernel_size=(filter_pixel, filter_pixel), activation='relu', padding="same")(second_input)
		norm1 = BatchNormalization()(conv1)
		dropout1 = Dropout(droprate)(norm1)

		conv2 = Conv2D(32, kernel_size=(filter_pixel, filter_pixel), activation='relu', border_mode="same")(dropout1)
		norm2 = BatchNormalization()(conv2)
		pool1 = MaxPooling2D()(norm2)
		dropout2 = Dropout(droprate)(pool1)

		flatten1 = Flatten()(dropout2)
		dense1 = Dense(y_train.shape[1]*2,use_bias=False)(flatten1)
		norm3 = BatchNormalization()(dense1)
		act1 = Activation('relu')(norm3) #14
		dropout3 = Dropout(droprate)(act1) #15
		return second_input, dropout3

	first_input, first_model = first_model(trans_X_train, trans_y_train)
	second_input, second_model = second_model(state_X_train, state_y_train)
	combinedInput = concatenate([first_model, second_model])
	shared_dense1 = Dense(trans_y_train.shape[1]*4+state_y_train.shape[1]*4)(combinedInput)
	first_output = Dense(trans_y_train.shape[1], name='first_output')(shared_dense1)
	second_output = Dense(state_y_train.shape[1], name='second_output')(shared_dense1)

	model = Model(inputs=[first_input, second_input], outputs=[first_output, second_output])
	model.compile(loss='mae',
	              optimizer='rmsprop', loss_weights=[1.,1.])
	model.fit({'first_input': trans_X_train, 'second_input': state_X_train}, {'first_output': trans_y_train, 'second_output': state_y_train}, epochs=epochs, batch_size=batch_size)
	return model

def basic_ConvLSTM(X_train, y_train):
	from keras.models import Sequential
	from keras.layers import TimeDistributed
	from keras.layers.convolutional import Conv3D, Conv2D
	from keras.layers.convolutional_recurrent import ConvLSTM2D
	from keras.layers.normalization import BatchNormalization
	import numpy as np
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3], 1)
	y_train = y_train.squeeze(axis=1)
	print(y_train.shape)
	#y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
	frames = 5 # The number of frames for each sequence
	input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3], 1)

	batch_size = 16
	droprate = 0.25
	epochs = 2
	filter_pixel = 3

	seq = Sequential()

	seq.add(ConvLSTM2D(filters=32, kernel_size=(filter_pixel, filter_pixel),
	               input_shape=input_shape,
	               padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=32, kernel_size=(filter_pixel, filter_pixel),
	               padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=32, kernel_size=(filter_pixel, filter_pixel),
	               padding='same', return_sequences=True))
	seq.add(BatchNormalization())

	seq.add(ConvLSTM2D(filters=32, kernel_size=(filter_pixel, filter_pixel),
	               padding='same'))
	seq.add(BatchNormalization())

	seq.add(Conv2D(filters=1, kernel_size=(filter_pixel, filter_pixel),activation='relu', padding='same', data_format='channels_last'))
	seq.compile(loss='mae', optimizer='rmsprop')
	seq.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)
	return seq

def plot_history(history):
    # Plot the history of accuracy
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig("model/model_accuracy.png")

    # Plot the history of loss
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig("model/model_loss.png")

def basic_2d_CNN(X_train, y_train):
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
	droprate=0.25
	batch_size = 128
	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
	model.add(BatchNormalization())
	model.add(Dropout(droprate))#3

	model.add(Conv2D(32, kernel_size=3, activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(droprate))#3

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Dense(y_train.shape[1]))

	#compile model using accuracy to measure model performance
	model.compile(loss='mae', optimizer='adam')

	#train the model
	model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_split=0.2)
	#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

	#make predictions
	#predict first 4 images in the test set
	#model.predict(X_test[:4])
	return model


def basic_CNN_LSTM_encoder_decoder(X_train, y_train):
	#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, MaxPooling2D, Dropout, RepeatVector
	from keras.layers.normalization import BatchNormalization
	from keras.utils.training_utils import multi_gpu_model

	verbose, epochs, batch_size = 2, 20, 16
	filter_pixel = 3
	droprate=0.25
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]), padding="same"))
	model.add(BatchNormalization())
	model.add(Dropout(droprate))#3

	model.add(Conv2D(32, kernel_size=(filter_pixel, filter_pixel), activation='relu', border_mode="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(droprate))#3

	model.add(Flatten())
	model.add(RepeatVector(1))
	model.add(LSTM(6, activation='relu'))
	model.add(Dense(y_train.shape[1]))
	model.compile(loss='mae', optimizer='adam')
	#gpu_model = multi_gpu_model(model, gpus=2)
	#gpu_model.compile(loss='mae', optimizer='adam')
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

def advanced_CNN_LSTM(X_train, y_train):
	#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
	from keras.models import Sequential
	from keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, MaxPooling2D, Dropout, RepeatVector, Lambda
	from keras.layers.normalization import BatchNormalization
	from keras.utils.training_utils import multi_gpu_model

	verbose, epochs, batch_size = 1, 20, 16
	filter_pixel = 3
	droprate=0.25
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])

	#create model
	model = Sequential()
	#add model layers
	model.add(Conv2D(32, kernel_size=(filter_pixel, filter_pixel), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]), padding="same"))
	model.add(BatchNormalization())
	model.add(Dropout(droprate))#3

	model.add(Conv2D(32, kernel_size=(filter_pixel, filter_pixel), activation='relu', border_mode="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())

	model.add(Conv2D(1, kernel_size=(filter_pixel, filter_pixel), activation='relu', border_mode="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(droprate))#3

	#model.add(Flatten())
	#model.add(RepeatVector(1))
	model.add(Lambda(lambda x: x[:,:,0]))
	model.add(LSTM(6, activation='relu'))
	#model.add(TimeDistributed(Dense(y_train.shape[1], activation='relu')))
	#model.add(TimeDistributed(Dense(y_train.shape[1])))
	model.add(Dense(y_train.shape[1]))
	model.compile(loss='mae', optimizer='adam')
	#gpu_model = multi_gpu_model(model, gpus=2)
	#gpu_model.compile(loss='mae', optimizer='adam')
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)
	return model

def basic_ConvLSTM_encoder_decoder(X_train, y_train):
	# if required,
	#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
	pass

def basic_stacked_autoencoder(X_train, y_train):
	from keras.layers import Dense, Input, Flatten
	from keras.models import Sequential, Model
	#Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion
	#Is Joint Training Better for Deep Auto-Encoders?
	#source from https://blog.keras.io/building-autoencoders-in-keras.html
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
	#print(y_train.shape)
	#print(X_train.shape)

	input_img = Input(shape=(X_train.shape[1], X_train.shape[2]))
	encoded = Dense(128, activation='relu')(input_img)
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)

	decoded = Dense(64, activation='relu')(encoded)
	decoded = Dense(128, activation='relu')(decoded)
	decoded = Flatten()(decoded)
	decoded = Dense(y_train.shape[1])(decoded)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])

	autoencoder.fit(X_train, y_train,
	                epochs=10,
	                shuffle=True, verbose=1)
	return autoencoder

def search_based_method(training, test, dist_func):
	from sklearn.metrics import mean_absolute_error
	import numpy as np
	from statistics import mean
	actuals = list()
	preds = list()
	for i in range(test.shape[0]-1):
		current = test[i,:]
		dists = [dist_func(current, training[j,:]) for j in range(training.shape[0])]
		idx = np.argmin(dists)
		#in case the closest past is the last time window
		if idx == training.shape[0]-1:
			continue
		pred = training[idx+1,:]
		actual = test[i+1,:]
		actuals.append(actual)
		preds.append(pred)
	mae = mean_absolute_error(actuals, preds)
	mape = mean_absolute_percentage_error(actuals, preds)

	return round(mae,4), round(mape,4)





