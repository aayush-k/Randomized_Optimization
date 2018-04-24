Follow setup instructions for ABAGAIL (https://github.com/pushkar/ABAGAIL)

Use ant to recompile any changes

NN Optimization Problem
java -cp ABAGAIL.jar opt.test.AKNeuralNetTest

- backprop, grid searching, accuracy vs iterations all available via commenting labeled parts of main method
- redundancy is the variable that indicates how many times an individual test will be repeated/averaged

Traveling Salesman test
java -cp ABAGAIL.jar opt.test.AKTravelingSalesmanTest.java

Max K Coloring Test
java -cp ABAGAIL.jar opt.test.AKMaxKColoringTest

Continous Peaks Test
java -cp ABAGAIL.jar opt.test.AKContinuousPeaksTest

In all of the above optimization problems, uncomment for loops labeled as __ grid search to conduct grid search over the parameters defined in paramGrid variable

Dataset
ABAGAIL/src/opt/test/gym_tt.csv
 - first 15000 rows are train Data
 - last 5000 rows are held out test data