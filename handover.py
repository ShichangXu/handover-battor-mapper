import os,sys,argparse
import re
import logging
import json

from pycommons import pycommons
from pycommons import generic_logging
logger = logging.getLogger('handover')

import common
from common import Experiment, open_file
from common import *

KEY = 'handover'

class HandoverData(object):
	RRC = 1
	SERVING_CELL = 2
	NEIGHBOR_CELL = 3
	RRC_STATE_CHANGE = 4

	def __init__(self, type, os_timestamp, diag_timestamp, **kwargs):
		self.os_timestamp = os_timestamp
		self.diag_timestamp = diag_timestamp
		self.type = type
		for key, value in kwargs.iteritems():
			setattr(self, key, value)

	@classmethod
	def parse(cls, data):
		start_line = data[0]
		end_line = data[-1]

		if cls.START_PATTERN.match(start_line) and \
				cls.END_PATTERN.match(end_line):
			timestamps_line = data[1]
			try:
				os_timestamp, diag_timestamp = [float(x) for x in timestamps_line.split(' ')]
				data = data[2:-1]
				d = {'data' : data}
				return HandoverData(cls.TYPE, os_timestamp, diag_timestamp, **d)
			except:
				return None


class HandoverRRC(HandoverData):
	TYPE = HandoverData.RRC
	START_PATTERN = re.compile(r'\[start LTE RRC\]')
	END_PATTERN = re.compile(r'\[end LTE RRC\]')

class HandoverServingCell(HandoverData):
	TYPE = HandoverData.SERVING_CELL
	START_PATTERN = re.compile(r'\[start Serving cell meas\]')
	END_PATTERN = re.compile(r'\[end Serving cell meas\]')

class HandoverNeighborCell(HandoverData):
	TYPE = HandoverData.NEIGHBOR_CELL
	START_PATTERN = re.compile(r'\[start Neighbor cell meas\]')
	END_PATTERN = re.compile(r'\[end Neighbor cell meas\]')

class HandoverRRCStateChange(HandoverData):
	TYPE = HandoverData.RRC_STATE_CHANGE
	START_PATTERN = re.compile(r'\[start LTE RRC state change\]')
	END_PATTERN = re.compile(r'\[end LTE RRC state change\]')

class Handover(object):
	START_PATTERN = re.compile(r'\[start.*\]')
	END_PATTERN = re.compile(r'\[end.*\]')

	def __init__(self, path):
		self.path = path
		self.data = []
		self.process()

	def process(self, start=0, stop=None):
		assert os.path.exists(self.path), "'%s' does not exist!" % (file)

		data = []
#		cmdline = 'cat %s | diag_parser' % (file)
#		ret, stdout, stderr = pycommons.run(cmdline, log=False)
#		lines = stdout.split('\n')

		lines = []
		with open_file(self.path, 'rb') as f:
			for line in f:
				lines.append(line.strip())

		for idx in range(len(lines)):
			line = lines[idx]
			m = self.START_PATTERN.match(line)
			if m:
				obj_data = []
				while idx < len(lines):
					line = lines[idx]
					obj_data.append(lines[idx])
					m = self.END_PATTERN.match(line)
					if m:
						break
					idx += 1

				obj = self.parse(obj_data)
				if not obj:
					#logger.debug("Could not convert: \n%s\n" % ('\n'.join(obj_data)))
					pass
				else:
					if obj.os_timestamp > start:
						if not stop or (stop and obj.os_timestamp < stop):
							data.append(obj)
			idx += 1

		self.data = data

	def get_data(self, start_idx, start=0, end=None):
		data = []
		last_idx = start_idx
		for idx, obj in enumerate(self.data):
			if obj.diag_timestamp > start:
				if not end or (end and obj.diag_timestamp < end):
					data.append(obj)
					last_idx = start_idx + idx
		return data, last_idx

	def parse(self, data):
		classes = [HandoverRRC, HandoverServingCell, HandoverNeighborCell, HandoverRRCStateChange]
		for c in classes:
			f = getattr(c, 'parse')
			obj = f(data)
			if obj:
				return obj
		return None

def process(file_path, experiments):
	logger.info("Processing ...")
	handover = Handover(file_path)

	expt_with_no_handover = 0
	num_cellular_expts = 0
	last_idx = 0

	for expt in experiments:
		for measurement in expt.get_measurements():
			if measurement.interface == Experiment.WIFI_IFACE:
				continue
			op = measurement.operation
			start_time = measurement.results.get(op + 'Begin', 0)
			end_time = measurement.results.get(op + 'End', 0)

			handover_data, last_idx = handover.get_data(last_idx, start_time, end_time)
			setattr(measurement, KEY, handover_data)
			if len(handover_data) == 0:
				expt_with_no_handover += 1
			num_cellular_expts += 1
	logger.info("%d/%d cellular experiments with no handover data" % (expt_with_no_handover, num_cellular_expts))


def main(argv):
	generic_logging.init(level=logging.WARN)
	global logger
	logger = logging.getLogger()

	config = json.loads(open('config').read())

	battor_logfile_path = config['parse']['file_path']
	sample_rate = get_sample_rate(battor_logfile_path)
	down_sample = config['parse']['down_sample']

	loglines = parse(**config)
	start_edge = find_start_edge(loglines, sample_rate, down_sample, **config['edge'])

	logcat_lines = logcat_generator(**config['logcat'])
	logcat_time_offset, battor_first_edge_timestamp = get_edge_timestamps(loglines, start_edge, logcat_lines)

	h = Handover(config['handover']['dir'], logcat_time_offset)

if __name__ == '__main__':
	main(sys.argv)

