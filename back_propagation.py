import numpy as np

class MLBP():
    def __init__(self, n_input,n_hidden, n_out):
        self.n_input = n_input
        self.n_hidden= n_hidden # This is a list that includes the num of neurons in each layer  # This is the number of layers
        self.n_out = n_out
        self.n_neuron = [self.n_input] + self.n_hidden + [self.n_out]
        self.n_layers = len(self.n_neuron)
        self.build_net()


    def build_net(self):
        """
        Built a neural network, including the weights, derivatives, and activations
        :return:
        """
        # define a list of weights, size of each layer is n_l*n_(l+1)
        # define a list to store the deritives
        self.weights = []
        self.derivatives = []
        for i in range(self.n_layers -1):
            weight = np.random.rand(self.n_neuron[i],self.n_neuron[i+1])
            self.weights.append(weight)
            num = self.n_neuron[i]
            derivarive = np.zeros(shape= (self.n_neuron[i],self.n_neuron[i+1]))
            self.derivatives.append(derivarive)

        # define a list to contain the activations
        self.activations = []
        for i in range(self.n_layers ):
            activation = np.zeros(shape = self.n_neuron[i])  # horizontal vector
            self.activations.append(activation)

    def forward_propagation(self,input):
        """
        Compute the output of the nerual network.
        :param input: the input of the neural network
        :return:
        """
        a = input
        self.activations[0] = a

        for i in range(self.n_layers-1):
            weight = self.weights[i]
            z = np.matmul(a, weight)
            a = self._activation_function(z)
            self.activations[i+1] = a

        return a

    def back_propagation(self,error):
        """

        :param error: error is the derivative of the loss-function wrt output(which is also the last layer activation)
                      error size: 1*n_neurons
        :return:
        """

        for i in reversed(range(self.n_layers-1)):
            activation_next = self.activations[i + 1]
            # print('activation_next',activation_next.shape)
            delta= self._derivative_sigmoid(activation_next) * error.reshape((1,-1))  # delta size
            activation = self.activations[i].reshape((-1,1))
            # print('activation',activation.shape, 'delta',delta.shape)
            derivative = np.matmul(activation, delta)
            # print('derivative',derivative.shape)
            self.derivatives[i] = derivative

            weight = self.weights[i].T
            # print('weight',weight.shape)
            error = np.matmul(delta, weight)
            # print('error',error.shape)
            # print('The size of derivation[{}] is {}'.format(i, derivative.shape))

    def gradient_descent(self, lr):
        for i in range(self.n_layers-1):
            self.weights[i] -= lr* self.derivatives[i]

    def train(self,inputs, targets, epochs, lr):

        for i in range(epochs):
            sum_error = 0

            for (input,target) in zip(inputs,targets):
                output = mlbp.forward_propagation(input)
                error = output - target
                mlbp.back_propagation(error)
                mlbp.gradient_descent(lr)

                sum_error += self._mse(target, output)
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def _mse(self,target,output):
        return np.mean( (target - output)**2)


    def _activation_function(self,x):
        # For sigmoid activation
        return 1.0/(1.0+ np.exp(-x))

    def _derivative_sigmoid(self,a):
        # if sx is a sigmoid function
        return a*(1.0-a)


if __name__ == '__main__':
    # create a dataset to train a network for the sum operation
    inputs = np.array([[np.random.random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    mlbp = MLBP(2, [5] ,1)
    mlbp.train(inputs,targets,50,0.1)

    input = np.array([0.1, 0.2])
    target = np.array([0.3])
    output = mlbp.forward_propagation(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))