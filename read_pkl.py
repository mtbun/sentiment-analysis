import pickle

path_name = 'data.pkl'
a_file = open(path_name, "rb")
output = pickle.load(a_file)
print(output)