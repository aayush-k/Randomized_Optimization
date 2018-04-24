package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying wine from 3 vintners
 *
 * @author Noah Roberts (edited by Aayush Kumar)
 * @version 1.0
 */
public class AKNeuralNetTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 10, hiddenLayer1 = 50, hiddenLayer2 = 50, hiddenLayer3 = 50, outputLayer = 8;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    // private static String results = "";
    private static int redundancy = 3;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[redundancy * 3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[redundancy * 3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[redundancy * 3];
    private static String[] oaNames = { "Randomized Hill Climbing", "Simulated Annealing", "Standard Genetic Algorithm"};

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        backpropNeuralNet();

        // find optimal hyperparams for sa (t_0, cooling rate) and ga (popSize, toMate, toMutate) via gridsearch at 750 iterations
        double[][][] paramGrid = new double[][][]{
            {
                {10, 500, 2500, 125000, 1E11}, // initial temp
                {0.1, 0.25, 0.5, 0.75, 0.95} // cooling rate
            },{
                {250, 1000}, // pop size
                {0.05, 0.1, 0.25}, //to mate
                {0.03, 0.06, 0.12 }, // to mutate
            }
        };
        gridSearch(paramGrid);

        // use optimal hyperparams vs iterations
        int[] trainingIterations = new int[]{
            1, 10, 50, 100, 250, 500,
             750, 1000, 2000};
        accVsIterations(trainingIterations);
    }

    private static void backpropNeuralNet() {
        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        // double[][][] data = { { { 1, 1 }, { .1, .9 } }, { { 0, 1 }, { 0, 1 } }, { { 0, 0 }, { .9, .1 } } };
        // Instance[] patterns = new Instance[data.length];
        // for (int i = 0; i < patterns.length; i++) {
        //     patterns[i] = new Instance(data[i][0]);
        //     patterns[i].setLabel(new Instance(data[i][1]));
        // }
        BackPropagationNetwork net = factory.createClassificationNetwork(
                new int[] { inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer });

        DataSet trainset = new DataSet(Arrays.copyOfRange(instances, 0, 15000));
        // ConvergenceTrainer c_trainer = new ConvergenceTrainer(
        //     );
        FixedIterationTrainer trainer = new FixedIterationTrainer(
                new BatchBackPropagationTrainer(trainset, net, new SumOfSquaresError(), new RPROPUpdateRule()), 1000);
        System.out.println("Training with Backprop");
        trainer.train();
        // System.out.println("Convergence in " + trainer.getIterations() + " iterations");

        Instance actual, predicted;
        double correct = 0, incorrect = 0;
        for (int i = 0; i < 15000; i++) {
            net.setInputValues(instances[i].getData());
            net.run();
            // System.out.println("~~");
            // System.out.println(instances[i].getLabel());
            // System.out.println(network.getOutputValues());

            predicted = instances[i].getLabel();
            actual = new Instance(net.getOutputValues());

            if (predicted.getData().argMax() == actual.getData().argMax()) {
                correct++;
            } else {
                incorrect++;
            }
        }
        double accuracy = correct / (correct + incorrect);
        System.out.println("\nCorrectly classified " + correct + " instances." + "\nIncorrectly classified "
                + incorrect + " instances.\nPercent correctly classified: " + df.format(accuracy));
    }

    private static void gridSearch(double[][][] paramGrid) {
        System.out.println("GRID SEARCH OA PARAMS");
        String summary = "";
        // grid search sa
        for (double t0 : paramGrid[0][0]) {
            for (double coolingRate: paramGrid[0][1]) {
                BackPropagationNetwork net = factory.createClassificationNetwork(
                    new int[] { inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer });
                NeuralNetworkOptimizationProblem netProb = new NeuralNetworkOptimizationProblem(set, net, measure);
                OptimizationAlgorithm oa = new SimulatedAnnealing(t0, coolingRate, netProb);

                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                train(oa, net, "|| SA || t0: " + t0 + ", coolingRate: " + coolingRate, 2000, 200, false);

                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa.getOptimal();
                net.setWeights(optimalInstance.getData());

                Instance predicted;
                Instance actual;
                start = System.nanoTime();
                for (int j = 15000; j < instances.length; j++) { // test set with optimal instance
                    net.setInputValues(instances[j].getData());
                    net.run();

                    predicted = instances[j].getLabel();
                    actual = new Instance(net.getOutputValues());

                    if (predicted.getData().argMax() == actual.getData().argMax()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);
                double accuracy = correct / (correct + incorrect);

                String result = "\nResults for " + "|| SA || t0: " + t0 + ", coolingRate: " + coolingRate + ": \nCorrectly classified " + correct + " instances."
                        + "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(accuracy) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                System.out.println(result);

            }
        }

        // grid search ga
        for (double popSize : paramGrid[1][0]) {
            for (double toMate : paramGrid[1][1]) {
                for (double toMutate: paramGrid[1][2]) {
                    System.out.println("Creating problem");
                    BackPropagationNetwork net = factory.createClassificationNetwork(
                        new int[] { inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer });
                    NeuralNetworkOptimizationProblem netProb = new NeuralNetworkOptimizationProblem(set, net, measure);
                    OptimizationAlgorithm oa = new StandardGeneticAlgorithm((int) popSize, (int) (toMate * popSize),
                            (int) (toMutate * popSize), netProb);

                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                    train(oa, net, "|| GA || popSize: " + popSize + ", toMate: " + toMate + ", toMutate: " + toMutate, 1000, 200, false);

                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10, 9);

                    Instance optimalInstance = oa.getOptimal();
                    net.setWeights(optimalInstance.getData());

                    Instance predicted;
                    Instance actual;
                    start = System.nanoTime();
                    for (int j = 15000; j < instances.length; j++) { // test set with optimal instance
                        net.setInputValues(instances[j].getData());
                        net.run();

                        predicted = instances[j].getLabel();
                        actual = new Instance(net.getOutputValues());

                        if (predicted.getData().argMax() == actual.getData().argMax()) {
                            correct++;
                        } else {
                            incorrect++;
                        }
                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10, 9);
                    double accuracy = correct / (correct + incorrect);

                    String result = "\nResults for " + "|| GA || popSize: " + popSize + ", toMate: " + toMate
                            + ", toMutate: " + ": \nCorrectly classified " + correct
                            + " instances." + "\nIncorrectly classified " + incorrect
                            + " instances.\nPercent correctly classified: " + df.format(accuracy) + "%\nTraining time: "
                            + df.format(trainingTime) + " seconds\nTesting time: " + df.format(testingTime)
                            + " seconds\n";
                    System.out.println(result);
                }
            }
        }

    }

    private static void accVsIterations(int[] iterations) {
        System.out.println("Measuring Accuracy vs Iterations");

        for (int iter: iterations) {
            System.out.println("\n##################\n" + iter + " Iterations");

            String results = "";

            System.out.println("\t(Re) Initializing Networks and Optimization Problems");

            for (int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] { inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer });
                nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
            }
            System.out.println("Made nns");

            double[] avgTestAccuracy = new double[3];
            double[] avgTrainAccuracy = new double[3];

            for (int i = 0; i < oa.length; i++) {

                if (i < redundancy) {
                    // continue;
                    oa[i] = new RandomizedHillClimbing(nnop[i]);
                } else if (i < (redundancy * 2)) {
                    // continue;
                    oa[i] = new SimulatedAnnealing(1E11, .25, nnop[i]);
                } else {
                    // continue;
                    oa[i] = new StandardGeneticAlgorithm(250, (int) (0.05 * 250), (int) (0.03 * 250), nnop[i]);
                }
                System.out.println("\n================================\n\t" + oaNames[i / redundancy] + ": Trial " + (i % redundancy));
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                double training_acc = train(oa[i], networks[i], oaNames[i / redundancy] + "@ iter: " + iter, iter, 50, true); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                Instance predicted;
                Instance actual;
                start = System.nanoTime();
                for (int j = 15000; j < instances.length; j++) { // test set with optimal instance
                    networks[i].setInputValues(instances[j].getData());
                    networks[i].run();

                    predicted = instances[j].getLabel();
                    // System.out.println(predicted);
                    actual = new Instance(networks[i].getOutputValues());
                    // System.out.println(actual);

                    if (predicted.getData().argMax() == actual.getData().argMax()) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
                System.out.println("%%%%%%%%%%%%%%% correct testing: " + correct);
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);
                double accuracy = correct / (correct + incorrect);
                avgTestAccuracy[i / redundancy] += accuracy / (redundancy * 1.0);
                avgTrainAccuracy[i / redundancy] += training_acc / (redundancy * 1.0);

                String result = "\nResults for " + oaNames[i / redundancy] + "\nTraining Accuracy: " + training_acc + "\nTest Accuracy: "
                        + df.format(accuracy) + "%\nTraining time: " + df.format(trainingTime) + " seconds\nTesting time: "
                        + df.format(testingTime) + " seconds\n";
                System.out.println(result);
                results += result;
            }

            System.out.println("\n");
            for (int i = 0; i < avgTestAccuracy.length; i++) {
                System.out.println(oaNames[i] + "\n\ttraining Accuracy" + avgTrainAccuracy[i] + "\n\ttesting Accuracy"
                        + avgTestAccuracy[i]);
            }
        }
    }

    private static double train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iter, int printFreq, boolean tAcc) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        double trainingAccuracy = 0, correct = 0, incorrect = 0;

        for(int i = 0; i < iter; i++) {
            oa.train();
        }

        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        Instance predicted;
        Instance actual;

        if (tAcc) {
            for (int j = 0; j < 15000; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                predicted = instances[j].getLabel();
                actual = new Instance(network.getOutputValues());

                double trash = predicted.getData().argMax() == actual.getData().argMax() ? correct++ : incorrect++;
            }

            trainingAccuracy = ((float) correct / (correct + incorrect));
        }

        return trainingAccuracy;

    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[20000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/aayush/Dev/cs4641/Randomized Optimization/ABAGAIL/src/opt/test/filtered_gym_data.csv")));
            br.readLine();

            for (int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[10]; // 10 attributes
                attributes[i][1] = new double[1];

                // y label (buckets of 20)
                attributes[i][1][0] = Integer.parseInt(scan.next()) / 20;

                // X vector (day of month)
                attributes[i][0][attributes[i][0].length - 1] = Integer.parseInt(scan.next().substring(8,10));

                //  X vector (others)
                for(int j = 0; j < attributes[i][0].length - 1; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println("Extracted Attrs");

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            int c = (int) attributes[i][1][0];
            double[] classes = new double[8]; // 8 classes
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }

        // shuffle and filter
        int index;
        Random random = new Random();
        Instance temp;

        // shuffle
        for (int i = instances.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = instances[index];
            instances[index] = instances[i];
            instances[i] = temp;
        }

        // filter
        instances = Arrays.copyOfRange(instances, 0, 20000);
        System.out.println("Processed Data into Instances");

        return instances;
    }

}