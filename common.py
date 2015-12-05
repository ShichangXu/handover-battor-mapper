import os,sys,argparse
import re
import gzip
import copy
import gc
import datetime

import numpy as np
import math
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter

from matplotlib import pyplot as plt
import itertools
import shutil
import serde

from memory_profiler import profile

import logging
logger = logging.getLogger('common')

ACTION_WIFI_RSSI_CHANGED = 'android.net.wifi.RSSI_CHANGED'
ACTION_WIFI_SCAN_RESULTS = 'android.net.wifi.SCAN_RESULTS'

from pycommons.file_entry import FileEntry

from collections import namedtuple

DUMP_DIR = 'dumps'

class Measurement(object):

	def __init__(self, iface, op, **kwargs):
		self.interface = iface
		self.operation = op
		self.results = kwargs

	def process(self):
		# Time is in seconds
		# Current in mA
		# Voltage in mV
		self.invalid = {}
		if self.interface == Experiment.CELLULAR_IFACE:
			# Check if it was LTE at the beginning and at the end
			try:
				begin_lte = cellular_results['beginCellInfo']['cellInfo']['mCellIdentityLte']
				end_lte = cellular_results['endCellInfo']['cellInfo']['mCellIdentityLte']
			except Exception, e:
				# These cellular results are invalid as we do not know when the
				# switch occured
				self.invalid['cellinfo'] = True


		log_period = self.loglines[1].timestamp - self.loglines[0].timestamp
		for idx in range(len(self.loglines) - 1):
			period = self.loglines[idx + 1].timestamp - self.loglines[idx].timestamp
			#assert  period == log_period, \
			#	"log periods are not equal :%d != %d" % (period, log_period)

		self.time = self.loglines[-1].timestamp - self.loglines[0].timestamp
		if self.time >= 22.0:
			# Experiment took too long..
			self.invalid['duration'] = True
		try:
			log_duration = self.results['%sDurationUs' % (self.operation)] # In uS
		except:
			pass

		energy = 0
		for line in self.loglines:
			if line.current >= 1650 or line.voltage >= 7200:
				self.energy = 0
				self.invalid['battor'] = True
				break
			energy += (line.current * line.voltage * log_period)
		self.energy = energy    # in uJ

		size = self.results.get('size', 0)	# bytes
		self.bytes = size
		if self.invalid:
			self.speed = None
			self.epb = None
		else:
			if size != 0:
				self.epb = self.energy / (size * 8.0)	# in uJ
				self.speed = (size * 8) / float(log_duration)

	def energy_per_bit(self):
		if self.invalid:
			return None
		return self.epb
	def bpj(self):
		# If it was marked invalid, return None
		if self.invalid:
			return None
		return self.bytes / float(self.energy)

	def get_average_latency(self):
		if self.operation != Experiment.OPERATION_LATENCY:
			# TODO: Maybe add tcpdump stuff here?
			return None
		try:
			value = float(self.results['values']['mean_rtt_ms'])
		except Exception, e:
			raise e
		return value

	def get_cell_tower_id(self):
		assert self.interface == Experiment.CELLULAR_IFACE, "Asking for cell tower ID when interface is %s" % (self.interface)
		tower_id = self.tower_id
		if tower_id is not None and len(tower_id) > 1:
#			logger.warn("Warning! Measurement has multiple cell towers associated with it!")
			return None
		if not tower_id:
#			logger.warn("Warning! Measurement has no cell tower associated with it!")
			return None
		return tower_id[0]

	def get_average_signal_strength(self, binsize=1):
		import handover
		if self.interface == Experiment.CELLULAR_IFACE:
			tower_id = self.get_cell_tower_id()
			if not tower_id:
				return None
			handover_data = getattr(self, handover.KEY, [])
			handover_data = [l.data for l in handover_data if l.type == handover.HandoverData.SERVING_CELL or l.type == handover.HandoverData.NEIGHBOR_CELL]
			handover_data = [x for x in handover_data if int(x[0].split(' ')[0]) == tower_id]
			rsrp_list = []
			for o in handover_data:
				id, rsrp, rsrq, rssi = [int(x) for x in o[0].split(' ')]
				rsrp_list.append(rsrp)
			if len(rsrp_list) == 0:
				return None
			# Normalize RSRP
			rsrp_list = [int((x / 16.0) - 180) for x in rsrp_list]
			avg_rsrp = int(math.ceil(np.mean(rsrp_list) / float(binsize)) * int(binsize))
			return avg_rsrp

		elif self.interface == 'wifi':
			import device2
			# Get SSID on which experiment was performed
			try:
				#XXX: For some reason the SSID is double-quoted '"SSID"'
				expt_ssid = self.results['properties']['ssid'].strip('"')
			except:
				logger.error("Failed to acquire experiment SSID!")
				return None
			rssi = []
			try:
				wifi_scans = getattr(self, device2.AllYouCanMeasure.KEY_ACTION_WIFI_SCAN_RESULTS, [])
				for line in wifi_scans:
						results = line['results']
						for entry in results:
							if entry['SSID'] == expt_ssid:
								rssi.append(entry['RSSI'])
				if len(rssi) == 0:
					return None
				avg_rssi = int(math.ceil(np.mean(rssi) / float(binsize)) * int(binsize))
				return avg_rssi
			except Exception, e:
				logger.error(e)
				return None


class Interface(object):

	def __init__(self, name):
		self.name = name

	def get_measurements(self):
		measurements = []
		for key in Experiment.OPERATIONS:
			m = getattr(self, key)
			if m:
				measurements.append(m)
		return measurements

class Experiment(dict):
	POWER_SYNC_KEY = 'powerSync'
	WIFI_IFACE = 'wifi'
	CELLULAR_IFACE = 'cellular'
	WIFI_INTERFACE_NAME = 'Wi-Fi'
	CELLULAR_INTERFACE_NAME = 'LTE'
	INTERFACE_NAMES = {WIFI_IFACE : WIFI_INTERFACE_NAME, CELLULAR_IFACE : CELLULAR_INTERFACE_NAME}
	INTERFACES = [WIFI_IFACE, CELLULAR_IFACE]
	WIFI_RESULT_KEY = 'wifiResults'
	CELLULAR_RESULT_KEY = 'cellularResults'
	INTERFACE_KEYS = [WIFI_RESULT_KEY, CELLULAR_RESULT_KEY]
	OPERATION_UPLOAD = 'upload'
	OPERATION_DOWNLOAD = 'download'
	OPERATION_LATENCY = 'latency'
	OPERATIONS = [OPERATION_UPLOAD, OPERATION_DOWNLOAD, OPERATION_LATENCY]

	def __init__(self, num):
		self.num = num

	def get_measurements(self):
		measurements = []
		for iface in Experiment.INTERFACES:
			obj = getattr(self, iface)
			if obj:
				measurements.extend(obj.get_measurements())
		return measurements

	def remove_measurement(self, measurement):
		for iface in Experiment.INTERFACES:
			iface_obj = getattr(self, iface)
			for op in Experiment.OPERATIONS:
				op_m = getattr(iface_obj, op)
				if op_m == measurement:
					setattr(iface_obj, op, None)


MEASUREMENT_COLORS = {
	Experiment.WIFI_IFACE : {
		Experiment.OPERATION_UPLOAD : 'c',
		Experiment.OPERATION_DOWNLOAD : 'm',
		Experiment.OPERATION_LATENCY : '#751D3C',
	},
	Experiment.CELLULAR_IFACE : {
		Experiment.OPERATION_UPLOAD : '#FF6800',   # Vivid Orange
		Experiment.OPERATION_DOWNLOAD : '#817066', # Medium Gray
		Experiment.OPERATION_LATENCY : 'k',        # Black
	},
}


BattorLogLine = namedtuple("BattorLogLine", ['timestamp', 'current', 'voltage'])
def parse_logline(line, sample_rate):
	tokens = line.split(' ')
	timestamp = float(tokens[0]) / sample_rate
	current = float(tokens[1])
	try:
		voltage = float(tokens[2])
	except:
		voltage = 0.0

	current = current if current > 0 else 0.0
	voltage = voltage if voltage > 0 else 0.0
	logline = BattorLogLine(timestamp, current, voltage)
	return logline


def split_experiments_to_measurements(experiments):
	power_syncs = []
	wifi_uploads = []
	wifi_downloads = []
	wifi_latency = []
	cellular_uploads = []
	cellular_downloads = []
	cellular_latency = []

	list_dict = {
		Experiment.POWER_SYNC_KEY : power_syncs,
		Experiment.WIFI_IFACE : {
			Experiment.OPERATION_UPLOAD : wifi_uploads,
			Experiment.OPERATION_DOWNLOAD : wifi_downloads,
			Experiment.OPERATION_LATENCY : wifi_latency,
		},
		Experiment.CELLULAR_IFACE : {
			Experiment.OPERATION_UPLOAD : cellular_uploads,
			Experiment.OPERATION_DOWNLOAD : cellular_downloads,
			Experiment.OPERATION_LATENCY : cellular_latency,
		},
	}

	for expt in experiments:
		if expt.powerSync:
			power_syncs.append(expt.powerSync)
		for measurement in expt.get_measurements():
			op_list = list_dict[measurement.interface][measurement.operation]
			op_list.append(measurement)
	metadata = get_metadata(experiments)
	return metadata, power_syncs, wifi_uploads, wifi_downloads, wifi_latency, cellular_uploads, cellular_downloads, cellular_latency


def get_experiment_directory(date, overwrite=False, light=False):
	directory = os.path.join(DUMP_DIR, str(date))
	if light:
		directory = os.path.join(directory, 'light')
	if os.path.exists(directory):
		if not overwrite:
			logger.warn("folder '%s' already exists!" % (directory))
		else:
			shutil.rmtree(directory)
	os.makedirs(directory)

	return directory

def serialize_experiments(date, experiments, overwrite=False, light=False):
	directory = get_experiment_directory(date, overwrite, light)
	logger.info("Serializing to '%s'" % (directory))
	serialize_metadata(directory, experiments)

	for expt in experiments:
		logger.info("Serializing: %d/%d" % (expt.num, experiments[-1].num))
		if expt.powerSync:
			serialize_power_sync(directory, expt)

		expt_num = expt.num
		for measurement in expt.get_measurements():
			serialize_measurement(directory, measurement, light)


def serialize_metadata(dumpdir, experiments):
	filename = os.path.join(dumpdir, 'metadata.bin')
	d = get_metadata(experiments)
	serde.serialize(d, filename)

def deserialize_metadata(dumpdir):
	filename = os.path.join(dumpdir, 'metadata.bin')
	d = serde.deserialize(filename)
	return d

def get_metadata(experiments):
	d = {}
	d['num_experiments'] = len(experiments)
	return d


def serialize_power_sync(dumpdir, experiment):
	ps_num = experiment.num
	ps = experiment.powerSync

	dirname = os.path.join(dumpdir, 'power_sync')
	if not os.path.exists(dirname):
		os.makedirs(dirname)


	filename = os.path.join(dirname, '%d.bin' % (ps_num))
	serde.serialize(ps, filename)

def deserialize_power_sync(dumpdir):
	dirname = os.path.join(dumpdir, 'power_sync')
	return _common_deserialize(dirname)

def serialize_measurement(dumpdir, measurement, light=False):
	m_num = measurement.num

	iface = measurement.interface
	op = measurement.operation

	dirname = os.path.join(dumpdir, iface, op)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	filename = os.path.join(dirname, '%d.bin' % (m_num))
	if light:
		loglines = measurement.loglines
		del measurement.loglines
		serde.serialize(measurement, filename)
		measurement.loglines = loglines
	else:
		serde.serialize(measurement, filename)

def deserialize_experiments(date, light=False):
	date = str(date)
	directory = os.path.join(DUMP_DIR, date)
	if light:
		directory = os.path.join(directory, 'light')

	power_syncs = deserialize_power_sync(directory)
	metadata = deserialize_metadata(directory)

	wifi_uploads = deserialize_measurements(directory, Experiment.WIFI_IFACE, Experiment.OPERATION_UPLOAD)
	wifi_downloads = deserialize_measurements(directory, Experiment.WIFI_IFACE, Experiment.OPERATION_DOWNLOAD)
	wifi_latency = deserialize_measurements(directory, Experiment.WIFI_IFACE, Experiment.OPERATION_LATENCY)

	cellular_uploads = deserialize_measurements(directory, Experiment.CELLULAR_IFACE, Experiment.OPERATION_UPLOAD)
	cellular_downloads = deserialize_measurements(directory, Experiment.CELLULAR_IFACE, Experiment.OPERATION_DOWNLOAD)
	cellular_latency = deserialize_measurements(directory, Experiment.CELLULAR_IFACE, Experiment.OPERATION_LATENCY)

	return metadata, power_syncs, wifi_uploads, wifi_downloads, wifi_latency, cellular_uploads, cellular_downloads, cellular_latency

def deserialize_measurements(dumpdir, iface, op):
	dirname = os.path.join(dumpdir, iface, op)
	return _common_deserialize(dirname)

def _common_deserialize(dirname):
	if not os.path.exists(dirname):
		raise Exception("'%s' does not exist! Not serialized" % (dirname))

	file_entry = FileEntry(dirname, None)
	file_entry.build(regex=['*.bin'])

	ret = []
	for entry in file_entry:
		measurement = serde.deserialize(entry.path())
		ret.append(measurement)
	return ret


def setup_parser(parser=None):
	if parser is None:
		parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, action='store', default='config',
			help='File containing config')
	return parser


def list_range(l):
	max_v = max(l)
	min_v = min(l)

	return max_v, min_v

def get_bin_width(l, count=100):
	maxv, minv = list_range(l)
	return (maxv - minv) / float(count)


def pad(string, length):
	retval = string
	padding = []
	while len(retval) + len(padding) < length:
		padding.append(' ')
	return retval + ''.join(padding)


def get_threshold(base, threshold):
	max_threshold = ((100 + threshold) * base) / 100.0
	min_threshold = ((100 - threshold) * base) / 100.0

	return min_threshold, max_threshold


def merge_overlap(list):
	# Code borrowed from http://codereview.stackexchange.com/a/69249
	result = []
	for higher in sorted(list, key=lambda tup: tup[0]):
		if not result:
			result.append(higher)
		else:
			lower = result[-1]
			# test for intersection between lower and higher:
			# we know via sorting that lower[0] <= higher[0]
			if higher[0] <= lower[1]:
				upper_bound = max(lower[1], higher[1])
				result[-1] = (lower[0], upper_bound)  # replace by result interval
			else:
				result.append(higher)
	return result

def find_start_edge(loglines, sample_rate, down_sample=0, **kwargs):
	edges = find_edges(loglines, down_sample, **kwargs)

	start_edge = edges[0]
	return start_edge


def _find_first_battor_power_sync(loglines, offset, down_sample, edge_samples, history_samples, current_threshold,
		min_current=None, stabilization_samples=0, drop_current=200, before_percentile=0, after_percentile=97):
	real_history_samples = history_samples
	real_history_downsampled = real_history_samples / down_sample

	history_samples_downsampled = (history_samples - stabilization_samples) / down_sample

	edge_samples_downsampled = edge_samples / down_sample

	result = []
	size = history_samples_downsampled + edge_samples_downsampled

	idx = size
	length = len(loglines)

	while idx < len(loglines) - size:
		start = idx
		end = idx + edge_samples_downsampled
		edge = (start, end)
		offsetted_edge = (start + offset, end + offset)
		if loglines[start].current > min_current and loglines[start].current - loglines[end].current > drop_current:
			if loglines[end].current < 550:	#FIXME: This should be configurable
				before_edge_samples = loglines[end-history_samples_downsampled:end]
				# Make sure the standard deviation of all samples before the edge is < threshold
				if np.std([x.current for x in before_edge_samples]) > current_threshold:
					logger.debug(str(offsetted_edge) + ' Failed std dev: %f < %d' % (np.std([x.current for x in before_edge_samples]), current_threshold))
					idx += 1
					continue
				after_edge_samples = loglines[end:end+history_samples_downsampled]
				# Make sure the standard deviation of all samples after the edge is < threshold
				#if np.std([x.current for x in after_edge_samples]) > current_threshold:
					#logger.debug(str(edge) + ' Failed std dev: %f < %d' % (np.std([x.current for x in after_edge_samples]), current_threshold))
					#continue

				after_edge_currents = [x.current for x in after_edge_samples]
				after_max_val = np.percentile(after_edge_currents, after_percentile)
				# Make sure all values before edge are > max(after edge)
				before_edge_currents = [x.current for x in before_edge_samples]
				before_min_val = np.percentile(before_edge_currents, before_percentile)
				if before_min_val > after_max_val:
					logger.debug("Found edge: %s" % (str(offsetted_edge)))
					return edge
				else:
					logger.debug(str(offsetted_edge) + ' Failed all(before_edge) > all(after_edge)')
					idx += 1
					continue
		idx += 1
	return None

def find_battor_power_syncs(date, loglines, **kwargs):
	hacks = kwargs.get('hacks', None)
	if hacks:
		hack = None
		for h in hacks:
			if h.get('date', None) == date:
				hack = h
				break
		if hack:
			logger.info("Applying hack ...")
			for k, v in hack.iteritems():
				if kwargs.get(k, None) is not None:
					kwargs[k] = v
		new_kwargs = dict(kwargs)
		del new_kwargs['hacks']
	else:
		new_kwargs = kwargs
	return _find_battor_power_syncs(date, loglines, **new_kwargs)

def _find_battor_power_syncs(date, loglines, sample_rate, down_sample,
		logcat_power_syncs, edge_samples, history_samples,
		current_threshold, min_current=None, stabilization_samples=0,
		drop_current=200, **kwargs):

	logger.info("Finding battor power syncs")
	power_syncs = []
	first_battor_power_sync_edge = _find_first_battor_power_sync(loglines, 0, down_sample, edge_samples, history_samples, current_threshold, min_current, stabilization_samples, drop_current, **kwargs)
	assert first_battor_power_sync_edge, "Did not find first power sync"
	logger.debug("Found 1 battor power sync")
	power_syncs.append(first_battor_power_sync_edge)

	battor_logline_time_diff_s = loglines[1].timestamp - loglines[0].timestamp

	first_logcat_power_sync_edge = logcat_power_syncs[0]['powerSyncEnd']

	for logcat_ps in logcat_power_syncs[1:]:
		time_diff_s = (logcat_ps['powerSyncEnd'] - first_logcat_power_sync_edge) / 1000.0

		edge_loc = int(first_battor_power_sync_edge[0] + ((time_diff_s * sample_rate) / float(down_sample)))
		sublist_start_idx = int(edge_loc - ((50 * sample_rate) / float(down_sample)))
		if sublist_start_idx < 0:
			sublist_start_idx = 0
		sublist_end_idx = int(edge_loc + ((50 * sample_rate) / down_sample))
		if sublist_end_idx > len(loglines):
			sublist_end_idx = len(loglines)

		bps = _find_first_battor_power_sync(loglines[sublist_start_idx:sublist_end_idx], sublist_start_idx, down_sample, edge_samples, history_samples, current_threshold, min_current, stabilization_samples, drop_current, **kwargs)
		# The result of the above operation is relative indices..make them absolute
		if not bps:
			power_syncs.append(None)
			logger.info("Did not find power sync")
			continue
		bps = (sublist_start_idx + bps[0], sublist_start_idx + bps[1])

		power_syncs.append(bps)
		logger.info("Found %d/%d power syncs" % (len(power_syncs), len(logcat_power_syncs)))
	return power_syncs

def smoothen(loglines, down_sample, key=None, num_samples=100, **kwargs):
	num_samples /= down_sample
	if key is not None:
		values = uniform_filter([getattr(x, key) for x in loglines], num_samples)
		for idx, v in itertools.izip(xrange(len(loglines)), values):
			try:
				loglines[idx] = loglines[idx]._replace(**{key : v})
			except:
				l = loglines[idx]
				setattr(l, key, v)
		return loglines
	else:
		values = uniform_filter(loglines, num_samples)
		return values

def custom_smoothen(loglines, key=None, num_samples=100, **kwargs):
	values = {}
	for idx in range(num_samples, len(loglines) - num_samples):
		start = idx - num_samples
		end = idx + num_samples

		if key:
			sublist = [getattr(x, key) for x in loglines[start:end]]
		else:
			sublist = [x for x in loglines[start:end]]

		median = np.median(sublist)

		values[idx] = median

	for idx, value in values.iteritems():
		if key:
			setattr(loglines[idx], key, value)
		else:
			loglines[idx] = value
	return loglines

def open_file(file_path, mode):
	# Handle compressed files
	filename, ext = os.path.splitext(file_path)
	if re.match('.tgz', ext) or re.match('.gz', ext):
		_open_fn = gzip.open
	else:
		_open_fn = open

	return _open_fn(file_path, mode)


def battor_parse(file_path, down_sample=1, start_timestamp=0, stop_timestamp=None, **kwargs):
	loglines = []

	sample_rate = float(get_sample_rate(file_path))

	down_sample_list = []

	num_lines = 0
	if not stop_timestamp:
		with open_file(file_path, 'rb') as f:
			for line in f:
				if line[0] == '#':
					continue
				num_lines += 1
	else:
		num_lines = int((1 + (float(stop_timestamp) - start_timestamp)) * sample_rate)

	loglines = [None for x in xrange(int(num_lines / float(down_sample)))]
	logger.debug("Finished zero initializing the list")

	idx = 0
	down_sampled_idx = 0
	with open_file(file_path, 'rb') as f:
		for line in f:
			if line[0] == '#':
				continue
			try:
				logline = parse_logline(line, sample_rate)
				del line
			except Exception, e:
				logger.error(e)
				continue

			if idx % 4000000 == 0:
				logger.debug("%d/%d" % (idx, num_lines))
				gc.collect()

			if down_sample > 1:
				down_sample_list.append(logline)
				if len(down_sample_list) == down_sample:
					timestamp = down_sample_list[0].timestamp
					avg_current = np.average([x.current for x in down_sample_list])
					avg_voltage = np.average([x.voltage for x in down_sample_list])
					logline = BattorLogLine(timestamp, avg_current, avg_voltage)
					loglines[down_sampled_idx] = logline
					del down_sample_list[:]
					down_sampled_idx += 1
				else:
					idx += 1
					continue
			else:
				loglines[idx] = logline
			# Check whether this logline is within the specified range
			if start_timestamp != 0 and logline.timestamp < start_timestamp:
				continue
			if stop_timestamp > 0 and logline.timestamp > stop_timestamp:
				return loglines[:idx]
			idx += 1
	return loglines

def get_sample_rate(battor_log_path):
	sample_rate_pattern = re.compile('#.*sample_rate=(?P<sample_rate>\d+).*')
	sample_rate = None

	with open_file(battor_log_path, 'rb') as f:
		for line in f:
			m = sample_rate_pattern.match(line)
			if m:
				sample_rate = int(m.group('sample_rate'))
				return sample_rate

def nearest_battor_logline_to_logcat_timestamp(loglines, logcat_timestamp, battor_power_syncs, logcat_power_syncs, threshold=1):
	nearest_battor_power_sync_end = None
	nearest_logcat_power_sync_end = None

	assert len(battor_power_syncs) == len(logcat_power_syncs), "battor_power_syncs != logcat_power_syncs (%d != %d)" % (len(battor_power_syncs), len(logcat_power_syncs))

	# Optimizations
	if logcat_timestamp < logcat_power_syncs[0]['powerSyncEnd']:
		# This timestamp is even before the first power sync
		# Anchor off of the first power sync
		nearest_battor_power_sync_end = loglines[battor_power_syncs[0][1]]
		nearest_logcat_power_sync_end = logcat_power_syncs[0]['powerSyncEnd']
	else:
		last_valid_bps = battor_power_syncs[0]
		last_valid_lps = logcat_power_syncs[0]
		assert last_valid_bps, "Apparently, no battor power syncs were found!"

		# We now iterate over all power sync pairs until we find one that
		# encompasses the locat_timestamp specified
		for idx, (bps, lps) in enumerate(itertools.izip(battor_power_syncs, logcat_power_syncs)):
			# Keep track of the last valid power sync
			# This is done in case we have Nones in the list
			if not bps:
				bps = last_valid_bps
				lps = last_valid_lps
			else:
				last_valid_bps = bps
				last_valid_lps = lps

			if idx == len(battor_power_syncs) - 1:
				nearest_battor_power_sync_end = loglines[bps[1]]
				nearest_logcat_power_sync_end = lps['powerSyncEnd']
				break

			next_lps = logcat_power_syncs[idx + 1]

			# Check if the pair of power syncs encompass our timestamp
			if logcat_timestamp > lps['powerSyncEnd'] and logcat_timestamp < next_lps['powerSyncEnd']:
				nearest_battor_power_sync_end = loglines[bps[1]]
				nearest_logcat_power_sync_end = lps['powerSyncEnd']
				break

	assert nearest_battor_power_sync_end, "Did not find a battor powerSync"
	assert nearest_logcat_power_sync_end, "Did not find a logcat powerSync"

	battor_sync_time_s = nearest_battor_power_sync_end.timestamp
	logcat_sync_time_ms = nearest_logcat_power_sync_end

	time_diff_s = (logcat_timestamp - logcat_sync_time_ms) / 1000.0

	exact_battor_timestamp = battor_sync_time_s + time_diff_s
	nearest_battor_logline, nearest_battor_idx = _find_nearest_logline(loglines, exact_battor_timestamp, threshold)

	return nearest_battor_logline, nearest_battor_idx

def _find_nearest_logline(loglines, timestamp, threshold):
	# XXX: There could be a bug because the timestamp that comes in
	# Is offset from base_timestamp (see plot_loglines)
	# Whereas we're not doing the offset here. This works as long as
	# battor loglines[0].timestamp is 0. If it isn't this will probably
	# give incorrect values
	# This never happens unless you decide to pull a set of loglines
	# starting from the middle of a day


	# Short circuit if timestamp is greater than largest battor timestamp
	if timestamp > loglines[-1].timestamp:
		return None, None

	# Some optimizations
	# XXX: Remove these optimizations if ther eis no guarantee about
	# consistent periodicity between consecutive loglines
	timestamp_diff = loglines[1].timestamp - loglines[0].timestamp
	# Now, technically, 'timestamp' should be in index
	# (timestamp / timestamp_diff)
	# We offset it by -1000 just to be on the safe side
	min_idx = max([0, int(timestamp / timestamp_diff) - 1000])
	# FIXME: Why +1000? Maybe this should be a precise value
	max_idx = min([len(loglines), int(timestamp / timestamp_diff) + 1000])
	subset = loglines[min_idx:max_idx]

	nearest = loglines[0]
	nearest_idx = 0
	for idx, l in enumerate(subset):
		cur = abs(l.timestamp - timestamp)
		best = abs(nearest.timestamp - timestamp)
		if cur < best:
			nearest = l
			nearest_idx = min_idx + idx
		if best == 0:
			# We're not going to get better than this
			return nearest, nearest_idx

	# Check to see if it is within threshold
	best = abs(nearest.timestamp - timestamp) / float(timestamp_diff)
	if best < threshold:
		return nearest, nearest_idx
	else:
		logger.warn("Did not find battor logline near %f (%d). Best: %f (%d)" % (timestamp, threshold, nearest.timestamp, best))
		return None, None

def parse_battor_loglines(file_path, **kwargs):
	logger.info("Parsing battor logs ...")
	parse_config = kwargs['battor']
	loglines = battor_parse(file_path, **parse_config)
	if parse_config['smoothen'] != 0:
		logger.info("Smoothing loglines")
		loglines = smoothen(loglines, parse_config['down_sample'], key='current', num_samples=parse_config['smoothen'])

	return loglines


def get_date(string):
	pattern = re.compile(r'.*(?P<date>2015\d{4}(_\d+)?).*')
	date = None
	try:
		date = pattern.match(string).group('date')
	except:
		pass
	finally:
		return date

'''
Haversine distance function borrowed from
https://github.com/mapado/haversin
'''
def haversine(point1, point2, miles=False):
	AVG_EARTH_RADIUS = 6371  # in km
	from math import radians, cos, sin, asin, sqrt
	""" Calculate the great-circle distance bewteen two points on the Earth surface.

	:input: two 2-tuples, containing the latitude and longitude of each point
	in decimal degrees.

	Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

	:output: Returns the distance bewteen the two points.
	The default unit is kilometers. Miles can be returned
	if the ``miles`` parameter is set to True.

	"""
	# unpack latitude/longitude
	lat1, lng1 = point1
	lat2, lng2 = point2

	# convert all latitudes/longitudes from decimal degrees to radians
	lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

	# calculate haversine
	lat = lat2 - lat1
	lng = lng2 - lng1
	d = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lng / 2) ** 2
	h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
	if miles:
		return h * 0.621371  # in miles
	else:
		return h  # in kilometers

def timestamp_to_date(timestamp):
	return datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d:%H")

color_dict = {}
colors= ['r', '#751D3C', 'b', 'g', 'm', 'k', 'c', '#BFACDA', '#E96048', '#E88BDD', '#AD68FC', '#DAE480', '#A868FC', 'y']
def create_color_dict(users):
	global color_dict
	for idx, user in enumerate(users):
		color_dict[user] = colors[idx]

	return color_dict
