import os,sys,argparse
import json
import itertools
import datetime

import logging
import pycommons.pycommons as pycommons
from pycommons import generic_logging
generic_logging.init(level=logging.INFO)
logger = logging.getLogger('post_processing')

import device1, device2, server, handover

def get_files(config):
	battor_files = []
	dev1_files = []
	dev2_files = []
	server_files = []
	handover_files = []

	base_dir = config['base']

	lists = [battor_files, dev1_files, dev2_files, server_files, handover_files]
	list_labels = ['battor', 'device1', 'device2', 'server', 'handover']
	for opt, opt_list in itertools.izip(list_labels, lists):
		config_opt = config[opt]
		opt_dir = os.path.join(base_dir, config_opt['dir'])
		opt_prefix = config_opt['prefix']
		opt_suffix = config_opt['suffix']

		for date in config['dates']:
			filename = gen_file_from_config(config_opt, date)
			abspath = os.path.join(opt_dir, str(date), filename)
			if not os.path.exists(abspath):
				abspath = os.path.join(opt_dir, filename)
				if not os.path.exists(abspath):
					logger.warn("Could not find file: " + filename)
					opt_list.append(None)
				else:
					opt_list.append(abspath)
			else:
				opt_list.append(abspath)

	return lists

def gen_file_from_config(subconfig, date):
	date_str = str(date)
	filename = subconfig['prefix'] + date_str + subconfig['suffix']
	return filename

def merge_logs(date, dev1_file, dev2_file, server_file, handover_file):
	experiments = []

	device1.process(dev1_file, experiments)
	device2.process(dev2_file, experiments)
	server.process(server_file, experiments)
	handover.process(handover_file, experiments)

	return experiments

def get_schedule(args):
	config = json.loads(open(args.config).read())
	schedule = {}
	# Get schedule
	try:
		schedule_file = os.path.join(config['base'], config['schedule'])
		sched = json.loads(open(schedule_file, 'rb').read())
		for k, v in sched.iteritems():
			schedule[int(k)] = v
	except Exception, e:
		logger.error(e)
		raise e
	return schedule

def build_user_dict(args):
	config = json.loads(open(args.config).read())
	schedule = get_schedule(args)
	unique_users = []
	[unique_users.append(schedule[x]) for x in sorted(schedule.keys()) if schedule[x] not in unique_users]
	user_dict = {}
	for idx, user in enumerate(unique_users):
		user_dict[user] = 'User %d' % (idx + 1)
	for k, v in user_dict.iteritems():
		print '%s -> %s' % (k, v)
	return user_dict

def setup_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, action='store', default='config',
			help='File containing config')
	parser.add_argument('-s', '--serialize', type=str, nargs='?',
			const='dump.bin', action='store', default=None,
			help='Serialize all experiments')
	parser.add_argument('-d', '--deserialize', type=str, nargs='?',
			const=True, action='store', default=None,
			help='Deserialize experiments')
	parser.add_argument('--overwrite', action='store_true', default=False,
			help='Overwrite serializations if they exist')
	parser.add_argument('--separate', action='store_true', default=False,
			help='Plot each experiment separately along with combined plot')
	parser.add_argument('--replot', action='store_true', default=False,
			help='Re-plot in case outdirs already exist')
	parser.add_argument('--dummy', action='store_true', default=False,
			help='Do dummy plots..for testing')
	parser.add_argument('--light', action='store_true', default=False,
			help='Use the lightweight serialized objects')
	return parser
