import scipy.io

# Load the .mat file
noisy_mat_file = 'evaluate_SIDD/ValidationGTBlocksSrgb.mat'
mat_contents = scipy.io.loadmat(noisy_mat_file)

# Print the keys of the loaded .mat file
print(mat_contents.keys())
