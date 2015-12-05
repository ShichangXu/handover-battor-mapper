import os,sys,argparse,re
import json
import itertools
import logging
logger = logging.getLogger('server')

KEY = 'server'

import common
from common import Experiment

def process(file_path, experiments):
	logger.info("Processing ...")
	jsons = get_server_jsons(file_path)

	failed = 0
	total = 0
	for expt in experiments:
		for measurement in expt.get_measurements():
			if measurement.operation == Experiment.OPERATION_LATENCY:
				continue
			total += 1
			measurement_id = measurement.results.get('id', None)
			assert measurement_id, "Measurement does not have an id: \n%s\n\n" % (json.dumps(measurement.results))
			server_log = jsons.get(measurement_id, None)
			if not server_log:
				logger.debug("Server does not have logs for: %s" % (measurement_id))
				failed += 1
				pass
			setattr(measurement, KEY, server_log)

	logger.warn("Server did not have logs for %d/%d measurements" % (failed, total))
def get_server_jsons(file_path):
	jsons = {}
	with open(file_path, 'rb') as f:
		for line in f:
			try:
				obj = json.loads(line)
				id = obj.get('id', None)
				if id:
					jsons[id] = obj
			except Exception, e:
				pass

	return jsons
