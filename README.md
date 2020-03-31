# LT2212 V20 Assignment 3

Part 1 - creating the feature table

a3_features.py can be run with 4 parameters: 
  - inputdir, is the path to the directory containing the documents to use as samples. The documents are expected to be separated in folders by author name. The name of each folder will be used as class.
  - outputfile, name of the file that will contain the features extracted from the documents. Every line in the file represents a document with the following information: name of the file, the specified number of features, 1 if it's part of the test set or 0 if it's part of the training set and finally the class.
  - dims, number of features to consider. The dimensionality reduction is obtained using PCA.
  - test, optional parameter to change the size of the test set. By default the division is 80% training set and 20% test set.
  

Part 2 and 3 - design and train the basic model, augment the model

a3_model.py can be run with 5 different parameters:

  - featurefile, is the path to the file created with a3_features.py that will be used to create the samples
  - e, optional, can be used to specify a number of epochs. By default, only 1 training loop is used.
  - nl, non-linearity. Optional parameter that can be either 1 or 2 and it is used to add a non-linearity to the neural network. The choice of 1 will add ELU activation to the hidden layer, the choice of 2 will add LogSigmoid.
  - size, optional, specifies the input size of the hidden layer. If a size is not specified, the neural network will be created without the hidden layer.
  - plot, optional, is used to run the bonus part.
  
Some of the options can be combined, for example the number of epochs and non-linearity or non-linearity and bonus part (even if, in this case, it is not possible to modify the size of the hidden layer).

The file a3_module.py contains both the main function and the class PerceptronModel that extends nn.Module and realizes the perceptron, using the chosen parameters (hidden layer or not and input size).

When the main function is run, the program reads the features file and stores the information in a DataFrame. This DataFrame is then separated into test set and training set and used by the function sample_data(df) to create the samples: for every possible combination of two documents, the function creates a row containing the two documents and the value 0 if they are not from the same author, 1 if they are. 
Before returning the new DataFrame containing the samples, the function removes the first and the last two columns of each feature vector (they represent the name of the file, the value indicating if it belongs to the training or test set and the name of the author). To balance the training set, the resulting DataFrame is also filtered to only keep a number of negative examples equal to the number of positive examples.

The training loop and the tests are realized in the function train_and_test_model(model, epochs, X, X_test, thresholds).
This function prints the characteristics of the Perceptron and runs the forward loop to train the model, using an utility function of the class PerceptronModel called predict(self, row) that creates a tensor using one row of the DataFrame and returns the predicted value (the two documents are separated in two different columns and they have to be concatenated to create the tensor).

When the training is complete, the test set (X_test) is used by the same function to predict the class values, using a threshold of 0.5 for the positive class and calculate accuracy, precision, recall and F-measure.

For the training loop, I used a BCELoss criterion and a Adam optimizer (both from the PyTorch library), with a learning rate of 0.001.


I run all the tests using 2 epochs and tried different input sizes from the hidden layer between the input size (200) and the output size (1):

No Hidden layer

Accuracy: 0.9558138766883524
Precision: 0.9451414196794272
Recall: 0.9678015917572075
F-measure: 0.9563372927816213

Hidden layer linear - size = 2

Accuracy: 0.9538489481853307
Precision: 0.9430653169127603
Recall: 0.9660182953006836
F-measure: 0.9544038238796718

Hidden layer linear - size = 134

Accuracy: 0.9493081470228857
Precision: 0.9323461184022371
Recall: 0.9689244080446484
F-measure: 0.9502834008097166

Hidden layer ELU - size = 1

Accuracy: 0.9480862587100822
Precision: 0.9318565199401636
Recall: 0.9668769195204914
F-measure: 0.9490437601296596

Hidden layer ELU - size = 2

Accuracy: 0.9631452065651729
Precision: 0.9376774958649315
Recall: 0.9922393580132757
F-measure: 0.9641871510172646

Hidden layer ELU - size = 134

Accuracy: 0.9804167629866913
Precision: 0.9664603841344149
Recall: 0.9953766388164196
F-measure: 0.9807054076918071

Hidden layer ELU - size = 200

Accuracy: 0.9808460750965953
Precision: 0.9676269390114655
Recall: 0.9949803507149698
F-measure: 0.9811130287537856

Hidden layer LogSigmoid - size = 1

Accuracy: 0.947128562464912
Precision: 0.9379589196183082
Recall: 0.9575971731448764
F-measure: 0.9476763187136414

Hidden layer LogSigmoid - size = 2

Accuracy: 0.9730358970971896
Precision: 0.9563527462724608
Recall: 0.9913146857765596
F-measure: 0.9735199208678591

Hidden layer LogSigmoid - size = 134

Accuracy: 0.9808130510881411
Precision: 0.9673246669876424
Recall: 0.995244542782603
F-measure: 0.9810860082036591

Hidden layer LogSigmoid - size = 200

Accuracy: 0.9813414352234074
Precision: 0.9681989014165944
Recall: 0.9953766388164196
F-measure: 0.9815996873575197


Part Bonus - plotting

To run the bonus part, a3_model.py has to be run using the -plot option, specifying the name of the file where to save the figure. It is possible to specify which non-linear layer to use for the PerceptronModel but not the size. This will run the train_and_test_model funcion three times, with hidden-layer size equal inputsize // 2, inputsize * 2 // 3 and inputsize, and plot the precision-recall curve using matplotlib with 5 different thresholds: 0.0, 0.25, 0.5, 0.625, 1.0. An example output can be found in the file figure.png in which it is used ELU as non-linearity.

