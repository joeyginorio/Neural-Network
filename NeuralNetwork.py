# Joey Velez-Ginorio
# Neural Network Practice
# =======================

import numpy as np
import scipy.special 


# Generates a neural network of any depth
class NeuralNetwork:

	# Initialize the network
	def __init__(self, depth, iNodes, hNodes, oNodes, learningRate):

		# Set dimensions of network
		self.iNodes = iNodes
		self.depth = depth
		self.hNodes = hNodes
		self.oNodes = oNodes
		self.learningRate = learningRate
		
		# Initialize weights
		# Uses the sampling trick for better intial value
		self.w = list()

		# Weights for input->hidden
		self.w.append(np.random.normal(0.0, pow(self.hNodes, -.5),
			(self.hNodes, self.iNodes)))

		# Weights for hidden->hidden
		for i in range(self.depth-1):
			self.w.append(np.random.normal(0.0, pow(self.hNodes,-.5),
				(self.hNodes, self.hNodes)))

		# Weights for hidden->output
		self.w.append(np.random.normal(0.0, pow(self.oNodes, -.5),
			(self.oNodes, self.hNodes)))

		self.activationFunction = lambda x: scipy.special.expit(x)
		self.inverseActivationFunction = lambda x: scipy.special.logit(x)

	# Train the network
	def train(self, inputs_list, targets_list):

		##################### FEED FORWARD #############################
		# Initialize input/output/error/weightUpdate lists
		self.inputs = list()
		self.outputs = list()
		self.errors = np.empty([len(self.w),1]).tolist()
		self.wUpdate = np.empty([len(self.w),1]).tolist()

		# Initial input / target
		self.inputs.append(np.array(inputs_list, ndmin=2).T)
		self.outputs.append(self.inputs[0])
		self.targets = np.array(targets_list, ndmin=2).T


		# Calculate input/output for input->hidden
		self.inputs.append(np.dot(self.w[0], self.outputs[0]))
		self.outputs.append(self.activationFunction(self.inputs[1]))

		# Calculate input/output for hidden->hidden
		for i in xrange(1, self.depth):
			self.inputs.append(np.dot(self.w[i],self.outputs[i]))
			self.outputs.append(self.activationFunction(self.inputs[i+1]))

		# Calculate input/output for hidden->output 
		self.inputs.append(np.dot(self.w[-1], self.outputs[-1]))
		self.outputs.append(self.activationFunction(self.inputs[-1]))

		################## BACK PROPAGATE ##############################
		# Calculate initial error (from output layer)
		self.errors[-1] = self.targets - self.outputs[-1]
		self.wUpdate[-1] = self.learningRate * np.dot(self.errors[-1] * \
			self.outputs[-1] * (1 - self.outputs[-1]), self.outputs[-2].T)
		self.w[-1] += self.wUpdate[-1]

		# Calculate back-propagated error for rest of network
		for i in xrange(2, len(self.w) + 1):
			
			# Allows the loop to run even if only one hidden layer present
			if i > len(self.w):
				break

			self.errors[-i] = np.dot(self.w[-(i-1)].T, self.errors[-(i-1)])
			self.wUpdate[-i] = self.learningRate * np.dot(self.errors[-i] * 
				self.outputs[-i] * (1-self.outputs[-i]), self.outputs[-(i+1)].T)
			self.w[-i] += self.wUpdate[-i]


	# Query the network
	def query(self, inputs_list):
		
		# Initialize input/output lists
		self.inputs = list()
		self.outputs = list()

		# Initial input
		self.inputs.append(np.array(inputs_list, ndmin=2).T)
		self.outputs.append(self.inputs[0])

		# Calculate input/output for input->hidden
		self.inputs.append(np.dot(self.w[0], self.outputs[0]))
		self.outputs.append(self.activationFunction(self.inputs[1]))

		# Calculate input/output for hidden->hidden
		for i in xrange(1, self.depth):
			self.inputs.append(np.dot(self.w[i],self.outputs[i]))
			self.outputs.append(self.activationFunction(self.inputs[i+1]))

		# Calculate input/output for hidden->output 
		self.inputs.append(np.dot(self.w[-1], self.outputs[-1]))
		self.outputs.append(self.activationFunction(self.inputs[-1]))

		return self.outputs[-1]

	# Peek into the mind of the network!
	def backquery(self, targets_list):

		# Convert list to numpy array
		self.targets = np.array(targets_list, ndmin=2).T

		self.inputs = np.empty([len(self.inputs),1]).tolist()
		self.outputs = np.empty([len(self.inputs),1]).tolist()


		# Calculate output/input of output layer
		self.outputs[-1] = self.targets
		self.inputs[-1] = self.inverseActivationFunction(self.targets)

		# Calculate output/input for hidden<-output w/rescaling
		self.outputs[-2] = np.dot(self.w[-1].T, self.inputs[-1])
	
		self.outputs[-2] -= self.outputs[-2].min()		
		self.outputs[-2] /= self.outputs[-2].max()
		self.outputs[-2] *= .98
		self.outputs[-2] += .01
		
		self.inputs[-2] = self.inverseActivationFunction(self.outputs[-2])

		# Calculate output/input for hidden<-hidden w/rescaling
		for i in xrange(1, self.depth-1):
			self.outputs[-(i+2)] = np.dot(self.w[-(i+1)].T, self.inputs[-(i+1)])

			self.outputs[-(i+2)] -= self.outputs[-(i+2)].min()
			self.outputs[-(i+2)] /= self.outputs[-(i+2)].max()
			self.outputs[-(i+2)] *= .98
			self.outputs[-(i+2)] += .01

			self.inputs[-(i+2)] = self.inverseActivationFunction(self.outputs[-(i+2)])

		# Calculate output/input for input<-hidden w/rescaling for both
		self.outputs[0] = np.dot(self.w[0].T, self.inputs[1])

		self.outputs[0] -= self.outputs[0].min()
		self.outputs[0] /= self.outputs[0].max()
		self.outputs[0] *= .98
		self.outputs[0] += .01

		self.inputs[0] = self.inverseActivationFunction(self.outputs[0])

		self.inputs[0] -= self.inputs[0].min()
		self.inputs[0] /= self.inputs[0].max()
		self.inputs[0] *= .98
		self.inputs[0] += .01

		return self.inputs[0]		


# Test Script for MNIST digit classification

# Specify the network parameters
numH = 1
iNodes = 784
hNodes = 200
oNodes = 10
learningRate = .2
epochs = 5

# Instantiate the network
NN = NeuralNetwork(numH, iNodes, hNodes, oNodes, learningRate)

# Load train / test datasets
trainingFile = open("mnist_train.csv", 'r')
trainingData = trainingFile.readlines()
trainingFile.close()

testingFile = open("mnist_test.csv", 'r')
testingData = testingFile.readlines()
testingFile.close()

# Retrain over epochs
for i in range(epochs):

	# Train all images in MNIST training set
	for image in trainingData: 

		# Convert csv to vector form
		image = image.split(',')

		# Hold onto label index
		labelIndex = int(image[0])

		# Process rest of vector into scaled image pixel array
		image = np.array(image[1:], dtype='float64')
		image /= 255.0
		image *= .99
		image += .01

		# Generate targets vector
		targets = np.zeros(oNodes) + .01
		targets[labelIndex] = .99

		NN.train(image, targets)


# Keep track of network performance
scores = list()
answers = list()
finalResults = list()

# Test for all images in MNIST test set
for image in testingData:

	# Convert csv into vector form
	image = image.split(',')

	# Hold onto label index / info
	correctLabel = int(image[0])
	answers.append(correctLabel)

	# Scale and shift image
	image = np.array(image[1:], dtype='float')
	image /= 255.0
	image *= .99
	image += .01

	# Query the network
	results = NN.query(image)
	label = np.argmax(results)
	finalResults.append(label)

	if(label == correctLabel):
		scores.append(1)
	else:
		scores.append(0)


scores = np.array(scores)
print "Performance: {}".format(float(scores.sum())/scores.size)

# Notes: Add intermediate results
# Output the hidden layer as well









		

