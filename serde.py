import cPickle as pickle

def serialize(obj, filename='dump.bin'):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)


def deserialize(filename='dump.bin'):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
	return obj


