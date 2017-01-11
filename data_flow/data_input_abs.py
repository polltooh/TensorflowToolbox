from abc import ABCMeta, abstractmethod
from data_abs import DataAbs

class DataInputAbs(DataAbs):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def get_label(self):
		pass

	@abstractmethod
	def get_input(self):
		pass

	@abstractmethod
	def load_data(self):
		pass
