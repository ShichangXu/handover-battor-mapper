import os,sys,argparse
import json
import itertools
import re
import itertools
from multiprocessing.pool import ThreadPool

import logging
import pycommons.pycommons as pycommons
from pycommons import generic_logging
generic_logging.init(level=logging.INFO)
logger = logging.getLogger('handover_energy')

import common
from common import parse_battor_loglines, pad, get_sample_rate, find_start_edge, get_date, Experiment
from logcat import logcat_generator, logcat_lines_to_experiments, get_edge_timestamps
import device1, device2, server, handover
from handover import Handover
import post_processing

def process(args):
	# First, we read the config
	config = json.loads(open(args.config).read())

	# Get the list of files of each type
	battor_files, dev1_files, dev2_files, server_files, handover_files = post_processing.get_files(config)

	# Build schedule
	schedule = post_processing.get_schedule(args)
	user_dict = post_processing.build_user_dict(args)
	sorted_user_dict_by_value = pycommons.sort_dict(user_dict, 1)
	user_order = [x[0] for x in sorted_user_dict_by_value]
	# Create colors per user
	common.create_color_dict(user_order)

	overall_experiments = {}

	for date, b_f, d1_f, d2_f, s_f, h_f in itertools.izip(config['dates'], battor_files, dev1_files, dev2_files, server_files, handover_files):
		date_str = str(date)
		assert schedule[date], 'Schedule does not contain entry for date: %s' % (date_str)
		user = schedule[date]

		logger.info("Processing: %s (%s)" % (date_str, user))

		# Merge all the separate log files together
		experiments = post_processing.merge_logs(date, d1_f, d2_f, s_f, h_f)
		# Find all power syncs as reported by logcat
		logcat_power_syncs = device1.get_power_syncs(None, experiments=experiments)
		# Logcat power syncs contain all the power syncs sorted by 'powerSyncBegin' key
		# Once we find the BattOr power syncs, we can line these up

		# Get all the BattOr loglines
		loglines = common.parse_battor_loglines(b_f, **config)

		sample_rate = common.get_sample_rate(b_f)
		down_sample = config['battor']['down_sample']
		# Get the start and end of each BattOr power sync
		# This is only the start and end of the *edge* corresponding to 'powerSyncEnd' key
		# in the device1's logcat logs
		battor_power_syncs = common.find_battor_power_syncs(date, loglines, sample_rate=sample_rate, down_sample=down_sample, logcat_power_syncs=logcat_power_syncs, **config['edge'])
		# Now we have both logcat power syncs and BattOr power syncs
		# We now have a way to anchor BattOr loglines with android timestamps

		# Build a handover object for the day's file
		handover = Handover(h_f)
		for data in handover.data:
			os_timestamp = data.os_timestamp
			nearest_logline, logline_idx = common.nearest_battor_logline_to_logcat_timestamp(loglines, os_timestamp, battor_power_syncs, logcat_power_syncs)

def main(argv):
	parser = post_processing.setup_parser()
	args = parser.parse_args(argv[1:])

	process(args)


if __name__ == '__main__':
	main(sys.argv)

