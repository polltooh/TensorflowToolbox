from abc import ABCMeta, abstractmethod

class ModelAbs(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def model_infer(self, data_ph, model_params):
		pass

	@abstractmethod
	def model_loss(self, data_ph, model_params):
		pass

	@abstractmethod
	def model_mini(self, loss):
		pass
	
