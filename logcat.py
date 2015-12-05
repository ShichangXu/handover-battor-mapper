import os,sys,re
import json
import operator
import itertools

import common
from common import *

import logging
logger = logging.getLogger('logcat')


class LogcatLogLine(object):
	THREADTIME_PATTERN = re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}.\d+)\s+(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<priority>[VDIWE])\s+(?P<tag>.*?): (?P<message>.*)$')
	def __init__(self, **kwargs):
		for key, value in kwargs.iteritems():
			setattr(self, key, value)

def logcat_generator(file_path):
	with open(file_path) as f:
		for line in f:
			try:
				m = LogatLogLine.THREADTIME_PATTERN.match(line)
				if m:
					j = json.loads(m.group('message'))
			except Exception, e:
				#logger.warn('Failed to convert line :%s' % (line))
				continue
			yield j

def logcat_parse(file_path, tag_pattern='.*'):
	tag_pattern = re.compile(tag_pattern)

	lines = []
	with open(file_path, 'rb') as f:
		for line in f:
			try:
				m = LogcatLogLine.THREADTIME_PATTERN.match(line)
				if m and tag_pattern.match(m.group('tag')):
					j = json.loads(m.group('message'))
					lines.append(j)
			except Exception, e:
				#logger.warn('Failed to convert line :%s' % (line))
				continue

	return lines


def find_nearest_power_sync(logcat_power_syncs, timestamp):
	nearest_ps_timestamp = logcat_power_syncs[0]['powerSyncEnd']
	best_time_diff = timestamp - nearest_ps_timestamp

	for ps in logcat_power_syncs:
		ps_timestamp = ps[0]['powerSyncEnd']
		time_diff = timesetamp - ps_timestamp

		if time_diff < best_time_diff:
			best_time_diff = time_diff
			nearest_ps = ps

	return nearest_ps


def find_logcat_edge(loglines):
	edges_dbg = []
	edges = []
	pattern = re.compile('.*[pP]owerSync\d?End')
	for l in loglines:
		line_edges = []
		for key in l.keys():
			if pattern.match(key):
				line_edges.append({key : l[key]})
				edges.append(l[key])

		line_edges.sort(key=lambda x: operator.itemgetter(1))
		edges_dbg.extend(line_edges)
	edges.sort()

	if len(edges) != 1:
		logger.warn('Warning: Found more than 1 power sync in logcat')
		logger.warn('Returning only the first edge')
	return edges[0]

def get_edge_timestamps(loglines, start_edge, logcat_lines):
	logcat_edge = find_logcat_edge(logcat_lines)
	logcat_time_offset = logcat_edge
	battor_first_edge_timestamp = loglines[start_edge[0]].timestamp
	logger.info('logcat origin: %.2f, battor origin: %.2f' % (logcat_time_offset, battor_first_edge_timestamp))
	logger.info('battor start: %.2f, battor end: %.2f' % (loglines[0].timestamp, loglines[-1].timestamp))

	return logcat_time_offset, battor_first_edge_timestamp

def get_wifi_rssi_values(logcat_lines, start=0, end=None):
	rssi_values = []
	for line in logcat_lines:
		# If line between start and end
		if line['timestamp'] >= start:
			if not end or (end and line['timestamp'] < end):
				# Then
				try:
					if common.ACTION_WIFI_RSSI_CHANGED == line['action']:
						# RSSI CHANGED
						rssi_values.append(line)
					elif common.ACTION_WIFI_SCAN_RESULTS == line['action']:
						# SCAN_RESULTS
						rssi_values.append(line)
				except Exception, e:
					logger.error(e)
					raise e
	return rssi_values





def logcat_lines_to_experiments(loglines, start_edge, logcat_lines, names=['wifi', 'cellular'], handover=None):
	experiments = []
	colocated_experiments = []

	logcat_time_offset, battor_first_edge_timestamp = get_edge_timestamps(loglines, start_edge, logcat_lines)

	# Find all available scan results
	for line in logcat_lines:
		expts = {}
		for name in names:
			try:
				expt_start = line['%sBegin' % (name)]
				expt_end = line['%sEnd' % (name)]
			except Exception, e:
				logger.debug('Failed to find key :%s' % (str(e)))
				continue
			try:
				start_time = ((expt_start - logcat_time_offset) / 1e3)
				end_time = ((expt_end - logcat_time_offset) / 1e3)
				logger.info('%s exp start: %.2f, exp end: %.2f' % (name, start_time, end_time))

				start_logline = find_nearest_logline(loglines, (battor_first_edge_timestamp + start_time), 5)
				start_idx = loglines.index(start_logline)
				end_logline = find_nearest_logline(loglines, (battor_first_edge_timestamp + end_time), 5)
				end_idx = loglines.index(end_logline)

				expt_loglines = loglines[start_idx:end_idx]
			except Exception, e:
				logger.warn('Failed to convert edge to logline :%s' % (str(e)))
				continue
			try:
				expt = Experiment(name, loglines[start_idx:end_idx], line)
				expts[name] = expt
				experiments.append(expt)
			except Exception, e:
				logger.warn('Failed to convert to Experiment :%s' % (str(e)))
				continue
			# If cellular, get handover data if available from expt_start to expt_end
			if name == 'cellular' and handover:
				data = []
				try:
					data = handover.get_data(start=expt_start, end=expt_end)
				except Exception, e:
					logger.warn('Failed to acquire handover data: %s' % (str(e)))
				expt.handover = data
			# Try to get wifi RSSI values for this experiment
			if name == 'wifi':
				rssi_values = []
				try:
					rssi_values = get_wifi_rssi_values(logcat_lines, start=expt_start - 60000, end=expt_end + 60000)
					logger.debug('Found %d wifi RSSI values' % (len(rssi_values)))
				except Exception, e:
					logger.warn('Failed to acquire wifi RSSI data: %s' % (str(e)))
				expt.wifi_rssi = rssi_values
		# If expts has values for all names, then this logline has co-located experiments
		if len(expts.keys()) == len(names):
			colocated_experiments.append(expts.values())

	return experiments, colocated_experiments

