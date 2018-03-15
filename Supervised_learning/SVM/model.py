
import os
import sys
from svm import *
from svm import __all__ as svm_all


__all__ = ['evaluations', 'svm_load_model', 'svm_predict', 'svm_read_problem',
           'svm_save_model', 'svm_train'] + svm_all

sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path

def svm_read_problem(data_file_name):

	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)

def svm_load_model(model_file_name):

	model = libsvm.svm_load_model(model_file_name.encode())
	if not model:
		print("can't open model file %s" % model_file_name)
		return None
	model = toPyModel(model)
	return model

def svm_save_model(model_file_name, model):

	libsvm.svm_save_model(model_file_name.encode(), model)

def evaluations(ty, pv):

	if len(ty) != len(pv):
		raise ValueError("len(ty) must equal to len(pv)")
	total_correct = total_error = 0
	sumv = sumy = sumvv = sumyy = sumvy = 0
	for v, y in zip(pv, ty):
		if y == v:
			total_correct += 1
		total_error += (v-y)*(v-y)
		sumv += v
		sumy += y
		sumvv += v*v
		sumyy += y*y
		sumvy += v*y
	l = len(ty)
	ACC = 100.0*total_correct/l
	MSE = total_error/l
	try:
		SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
	except:
		SCC = float('nan')
	return (ACC, MSE, SCC)

def svm_train(arg1, arg2=None, arg3=None):

	prob, param = None, None
	if isinstance(arg1, (list, tuple)):
		assert isinstance(arg2, (list, tuple))
		y, x, options = arg1, arg2, arg3
		param = svm_parameter(options)
		prob = svm_problem(y, x, isKernel=(param.kernel_type == PRECOMPUTED))
	elif isinstance(arg1, svm_problem):
		prob = arg1
		if isinstance(arg2, svm_parameter):
			param = arg2
		else:
			param = svm_parameter(arg2)
	if prob == None or param == None:
		raise TypeError("Wrong types for the arguments")

	if param.kernel_type == PRECOMPUTED:
		for xi in prob.x_space:
			idx, val = xi[0].index, xi[0].value
			if xi[0].index != 0:
				raise ValueError('Wrong input format: first column must be 0:sample_serial_number')
			if val <= 0 or val > prob.n:
				raise ValueError('Wrong input format: sample_serial_number out of range')

	if param.gamma == 0 and prob.n > 0:
		param.gamma = 1.0 / prob.n
	libsvm.svm_set_print_string_function(param.print_func)
	err_msg = libsvm.svm_check_parameter(prob, param)
	if err_msg:
		raise ValueError('Error: %s' % err_msg)

	if param.cross_validation:
		l, nr_fold = prob.l, param.nr_fold
		target = (c_double * l)()
		libsvm.svm_cross_validation(prob, param, nr_fold, target)
		ACC, MSE, SCC = evaluations(prob.y[:l], target[:l])
		if param.svm_type in [EPSILON_SVR, NU_SVR]:
			print("Cross Validation Mean squared error = %g" % MSE)
			print("Cross Validation Squared correlation coefficient = %g" % SCC)
			return MSE
		else:
			print("Cross Validation Accuracy = %g%%" % ACC)
			return ACC
	else:
		m = libsvm.svm_train(prob, param)
		m = toPyModel(m)

		# If prob is destroyed, data including SVs pointed by m can remain.
		m.x_space = prob.x_space
		return m

def svm_predict(y, x, m, options=""):


	def info(s):
		print(s)

	predict_probability = 0
	argv = options.split()
	i = 0
	while i < len(argv):
		if argv[i] == '-b':
			i += 1
			predict_probability = int(argv[i])
		elif argv[i] == '-q':
			info = print_null
		else:
			raise ValueError("Wrong options")
		i+=1

	svm_type = m.get_svm_type()
	is_prob_model = m.is_probability_model()
	nr_class = m.get_nr_class()
	pred_labels = []
	pred_values = []

	if predict_probability:
		if not is_prob_model:
			raise ValueError("Model does not support probabiliy estimates")

		if svm_type in [NU_SVR, EPSILON_SVR]:
			info("Prob. model for test data: target value = predicted value + z,\n"
			"z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g" % m.get_svr_probability());
			nr_class = 0

		prob_estimates = (c_double * nr_class)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi, isKernel=(m.param.kernel_type == PRECOMPUTED))
			label = libsvm.svm_predict_probability(m, xi, prob_estimates)
			values = prob_estimates[:nr_class]
			pred_labels += [label]
			pred_values += [values]
	else:
		if is_prob_model:
			info("Model supports probability estimates, but disabled in predicton.")
		if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
			nr_classifier = 1
		else:
			nr_classifier = nr_class*(nr_class-1)//2
		dec_values = (c_double * nr_classifier)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi, isKernel=(m.param.kernel_type == PRECOMPUTED))
			label = libsvm.svm_predict_values(m, xi, dec_values)
			if(nr_class == 1):
				values = [1]
			else:
				values = dec_values[:nr_classifier]
			pred_labels += [label]
			pred_values += [values]

	ACC, MSE, SCC = evaluations(y, pred_labels)
	l = len(y)
	if svm_type in [EPSILON_SVR, NU_SVR]:
		info("Mean squared error = %g (regression)" % MSE)
		info("Squared correlation coefficient = %g (regression)" % SCC)
	else:
		info("Accuracy = %g%% (%d/%d) (classification)" % (ACC, int(l*ACC/100), l))

	return pred_labels, (ACC, MSE, SCC), pred_values


