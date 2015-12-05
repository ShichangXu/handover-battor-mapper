import os,sys,argparse,re
import json

import logcat
import itertools

from common import Experiment, Interface, Measurement
import common
import logging
from pycommons import generic_logging
generic_logging.init(level=logging.INFO)
logger = logging.getLogger('device1')


def process(file_path, experiments):
	logger.info("Processing ...")
	jsons = logcat.logcat_parse(file_path, tag_pattern='IFSelect->MeasurementTask')

	experiment_number = 0
	for j in jsons:
		# powerSyncs are included in experiment logs
		if not j.get('experimentDurationUs', None):
			logger.debug("Skipping line due to no experimentDurationUs: %s" % (j))
			continue
		# If the object has key experimentDurationUs, then we have an experiment
		expt = parse_experiment(j, experiment_number)
		experiments.append(expt)
		experiment_number += 1

def parse_experiment(expt_json, experiment_number):
	expt = Experiment(experiment_number)

	wifi_results = expt_json.get(Experiment.WIFI_RESULT_KEY, {})
	cellular_results = expt_json.get(Experiment.CELLULAR_RESULT_KEY, {})

	if cellular_results:
		try:
			begin_pci = cellular_results['beginCellInfo']['cellInfo']['mCellIdentityLte']['mPci']
			end_pci = cellular_results['endCellInfo']['cellInfo']['mCellIdentityLte']['mPci']
			pci = list(set([begin_pci, end_pci]))
			if begin_pci != end_pci:
				logger.warn("Cell ID changed between measurements")
		except Exception, e:
			logger.error(e)
			pci = None

	# Charging state
	is_charging = expt_json.get('isCharging', False)
	for iface_name, iface_key, iface_results in itertools.izip(Experiment.INTERFACES, Experiment.INTERFACE_KEYS, [wifi_results, cellular_results]):
		iface = Interface(iface_name)

		operations = Experiment.OPERATIONS
		op_dicts = [iface_results.get(x, None) for x in operations]

		for op, op_dict in itertools.izip(operations, op_dicts):
			measurement = None
			if op_dict is not None:
				measurement = Measurement(iface_name, op, **op_dict)
				measurement.num = expt.num
				measurement.is_charging = is_charging
				measurement.beginCellInfo = cellular_results.get('beginCellInfo', None)
				measurement.endCellInfo = cellular_results.get('endCellInfo', None)
				if iface_name == Experiment.CELLULAR_IFACE:
					measurement.tower_id = pci
#			# Filter logic
#			if measurement is not None and op != Experiment.OPERATION_LATENCY and \
#					measurement.results['%sDurationUs' % (op)] >= 22000000:
#				measurement = None
			setattr(iface, op, measurement)

		setattr(expt, iface_name, iface)

	expt.experimentDurationUs = expt_json['experimentDurationUs']
	expt.powerSync = expt_json.get('powerSync', None)

	return expt

def get_power_syncs(file_path, experiments=[]):
	power_syncs = []

	if len(experiments) == 0:
		process(file_path, experiments)

	for expt in experiments:
		ps = expt.powerSync
		if ps:
			power_syncs.append(ps)

	power_syncs = sorted(power_syncs, key=lambda x: x['powerSyncBegin'])
	return power_syncs

def assign_loglines_to_experiments(loglines, experiments, battor_power_syncs):
	logger.info("Assigning loglines to experiments")
	logcat_power_syncs = get_power_syncs(None, experiments=experiments)

	for idx, expt in enumerate(experiments):
		logger.info("\t%d/%d" % (idx, len(experiments)))
		for measurement in expt.get_measurements():
			op = measurement.operation
			start = measurement.results.get(op + 'Begin')
			end = measurement.results.get(op + 'End')

			start_logline, start_idx = common.nearest_battor_logline_to_logcat_timestamp(loglines, start, battor_power_syncs, logcat_power_syncs)
			end_logline, end_idx = common.nearest_battor_logline_to_logcat_timestamp(loglines, end, battor_power_syncs, logcat_power_syncs)

			# Filter out all measurements that have current or voltage readings above normal
#			measurement_loglines = loglines[start_idx:end_idx]
#			if not all([x.current <= 1650 for x in measurement_loglines]) or \
#					not all([x.voltage <= 7200 for x in measurement_loglines]):
#						iface_m = getattr(expt, measurement.interface)
#						logger.warn("Removing measurement due to high voltage/current")
#						setattr(iface_m, op, None)
#			else:
			if start_logline and end_logline:
				logger.info("\t\tAssigned %s" % (op))
				setattr(measurement, 'loglines', loglines[start_idx:end_idx])
				setattr(measurement, 'start_idx', start_idx)
				setattr(measurement, 'end_idx', end_idx)
			else:
				logger.info("\t\tNo loglines found for measurement. Removing it")
				expt.remove_measurement(measurement)
				continue
			measurement.start_logline_idx = start_idx
			measurement.end_logline_idx = end_idx
			measurement.process()

