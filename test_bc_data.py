import pickle

file = open('./bc_data/chair_ingolf_0650.gail.p0.123_step_00011468800_2_trajs.pkl', 'rb')

# dump information to that file
data = pickle.load(file)

# breakpoint()

# -> file.close()
# (Pdb) p data.keys()
# *** AttributeError: 'list' object has no attribute 'keys'
# (Pdb) p len(data)
# 2
# (Pdb) data[0].keys()
# dict_keys(['obs', 'ob_images', 'actions', 'rewards', 'dones'])
# (Pdb) len(data[0]['ob_images'])
# 17
# (Pdb) data[0]['ob_images'][0].shape
# (500, 500, 3)
# (Pdb)

# close the file
file.close()
