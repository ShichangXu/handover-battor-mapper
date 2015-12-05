import os,sys,argparse,re
import json
import itertools
import logging
logger = logging.getLogger('device2')

import logcat


import common
from common import Experiment

class AllYouCanMeasure(dict):
	ACTION_LOCATION_UPDATE = "edu.buffalo.cse.phonelab.allyoucanmeasure.receivers.LocationReceiver.LocationUpdated"
	ACTION_WIFI_SCAN_RESULTS = "android.net.wifi.SCAN_RESULTS"
	ACTION_CELLULAR_INFO = "edu.buffalo.cse.phonelab.allyoucanmeasure.receivers.CellularReceiver.CellInfo"
	ACTION_ACTIVITY_UPDATE = "edu.buffalo.cse.phonelab.allyoucanmeasure.services.ActivityIntentService.ActivityUpdate"

	KEY_ACTION_LOCATION_UPDATE = 'location'
	KEY_ACTION_WIFI_SCAN_RESULTS = 'wifiScan'
	KEY_ACTION_CELLULAR_INFO = 'cellularInfo'
	KEY_ACTION_ACTIVITY_UPDATE = 'activity'

	expt_key_dict = {
			ACTION_LOCATION_UPDATE : KEY_ACTION_LOCATION_UPDATE,
			ACTION_WIFI_SCAN_RESULTS : KEY_ACTION_WIFI_SCAN_RESULTS,
			ACTION_CELLULAR_INFO : KEY_ACTION_CELLULAR_INFO,
			ACTION_ACTIVITY_UPDATE : KEY_ACTION_ACTIVITY_UPDATE,
	}

	def __init__(self, **kwargs):
		for key, value in kwargs.iteritems():
			self[key] = value

	def get_key_from_action(self):
		return self.expt_key_dict[self['action']]

def process(file_path, experiments):
	logger.info("Processing ...")
	jsons = logcat.logcat_parse(file_path, tag_pattern='AllYouCanMeasure-.*')

	last_idx = 0
	for expt in experiments:
		for measurement in expt.get_measurements():
			op = measurement.operation
			start_time = measurement.results.get(op + 'Begin', 0)
			end_time = measurement.results.get(op + 'End', 0)

			last_idx = add_info(measurement, jsons, last_idx, start_time, end_time)


def add_info(measurement, info_list, start_idx, start_time, end_time):
	sublist = info_list
	last_idx = start_idx

	for idx, info in enumerate(sublist):
		timestamp = info['timestamp']
		if timestamp >= start_time and timestamp < end_time:
			aycm = AllYouCanMeasure(**info)
			key = aycm.get_key_from_action()
			if not getattr(measurement, key, None):
				setattr(measurement, key, [])
			aycm_list = getattr(measurement, key)
			aycm_list.append(aycm)
			last_idx = start_idx + idx

	return last_idx

