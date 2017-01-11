from abc import ABCMeta, abstractmethod
from data_abs import DataAbs

class DataPhAbs(DataAbs):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def get_label(self):
		pass

	@abstractmethod
	def get_input(self):
		pass

	def feed_data(self, data_input):
		label_ph = get_label()
		input_ph = get_input()
		feed_dict = dict()
		add_to_dict(feed_dict, self.get_label(), data_input.get_label())
		add_to_dict(feed_dict, self.get_input(), data_input.get_input())
		return feed_dict

	def add_to_dict(feed_dict, ph, value):
		if isinstance(feed_dict, list):
			assert(len(feed_dict) == len(value))
			for i in xrange(len(feed_dict)):
				feed_dict[ph[i]] = value[i]
			
