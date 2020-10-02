import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()
    training_inputs = np.array([[3,0.5,1], #temp/humidite/vent
                                [1.0,6.5,1.5],
                                [0.5,7.0,2.0],
                                [2.5,0.5,1.2]])

    training_outputs = np.array([[1,0,0,1]]).T
    neural_network.train(training_inputs, training_outputs, 15000)
    print("Respective weights after training : ")
    print(neural_network.synaptic_weights)
    user_input_one = str(float(input("Temperature (degrees): "))/10)
    user_input_two = str(float(input("Humidity (%): "))/10)
    user_input_three = str(float(input("Wind (km/h): "))/10)
    
    print("New Situation : ", user_input_one, user_input_two, user_input_three)
    print("Output data: ")
    outputfinal = float(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print(outputfinal)
    if (outputfinal >= 0.5 ) :
        percentfinal = outputfinal*100
        print("the weather as {} percent of chance to be good !".format(percentfinal))
    elif (outputfinal <= 0.5 ) :
        print("The weather as higher chance to be bad... Take umbrella")
