import ast
import sys
import numpy as np
import copy

filename = sys.argv[1]

if False:
	with open(filename, 'r') as f:
		for line in f:
			try:
				data = ast.literal_eval(" ".join(line.split(" ")[1:]))

				if isinstance(data, list):
					new_data = {}
					for key in data[0].keys():
						new_data[key] = np.mean([d[key] for d in data])
					print(new_data)
				else:
					print(data)
			except Exception as e:
				print(line, end='')
else:
	with open(filename, 'r') as f:
		best_args = {}
		best_value = {0:-100., 1:-100., 2:-100., 3:-100., 4:-100.}
		curr_args = None
		curr_fold = 0

		for line in f:
			items = line.split(" ")
			key = items[0]
			value = " ".join(items[1:])

			if key == "args:":
				curr_args = ast.literal_eval(value)
				curr_fold = curr_args['test_fold']

			text_key = "test_acc:" if not 'regression' in filename else "test_loss:"
			if key == text_key:
				values = ast.literal_eval(value)
				if isinstance(values, list):
					new_data = {}
					for key in values[0].keys():
						new_data[key] = np.mean([d[key] for d in values])
					values = new_data

				for k, v in values.items():
					if 'regression' in filename:
						v = -v
					if v > best_value[curr_fold]:
						best_value[curr_fold] = v
						best_args[curr_fold] = curr_args.copy()
						best_args[curr_fold]['epochs'] = k

		print(",\n".join(str(i) for i in best_args.values()))
