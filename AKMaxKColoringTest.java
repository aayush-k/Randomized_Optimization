package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 *
 * @author kmandal edited by Aayush Kumar
 * @version 1.0
 */
public class AKMaxKColoringTest {
    /** The n value */
    private static final int N = 50; // number of vertices
    private static final int L =4; // L adjacent nodes per vertex
    private static final int K = 8; // K possible colors
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random(N*L);
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;
            vertex.setAdjMatrixSize(L);
            for(int j = 0; j <L; j++ ){
            	 vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
            }
        }
        /*for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }*/
        // for rhc, sa, and ga we use a permutation based encoding
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        Distribution df = new DiscreteDependencyTree(.1);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double[][][] paramGrid = new double[][][] {
            {
                { 10, 500, 2500, 125000, 1E11 }, // initial temp
                { 0.1, 0.25, 0.5, 0.75, 0.95 } // cooling rate
            }, {
                { 10, 250, 1000 }, // pop size
                { 0.05, 0.1, 0.25 }, //to mate
                { 0.03, 0.06, 0.12 }, // to mutate
            }, {
                { 150, 200 }, // samples
                { 10, 50, 100 }, //to keep
            }, { //iterations
                { 1, 5, 10, 25, 500, 1000, 50000, 200000 }, // rhc, sa
                { 1, 5, 10, 25, 500, 1000} // ga, mimic
            }
        };

        FixedIterationTrainer fit;

        // RHC
        for (double iter : paramGrid[3][0]) {
            // long starttime = System.currentTimeMillis();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, (int) iter);
            fit.train();
            System.out.println(iter + "iter, RHC: " + ef.value(rhc.getOptimal()));
            System.out.println(ef.foundConflict());
            // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        }

        System.out.println("============================");
        double t0 = 500, coolingRate = 0.25;

        for (double iter : paramGrid[3][0]) {

        // // SA grid search
        // for (t0 : paramGrid[0][0]) {
        //     for (coolingRate : paramGrid[0][1]) {
                // starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(t0, coolingRate, hcp);
                fit = new FixedIterationTrainer(sa, (int) iter);
                fit.train();
                System.out.println( "\n" +
                        iter + "iter, SA || t0: " + t0 + ", coolingRate: " + coolingRate + "\n\t" + ef.value(sa.getOptimal()));
                System.out.println(ef.foundConflict());
                // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        //     }
        // }
        }

        System.out.println("============================");
        double popSize = 1000, toMate = 0.05, toMutate = 0.03;

        for (double iter : paramGrid[3][1]) {
            // GA grid search
            // for (popSize : paramGrid[1][0]) {
            //     for (toMate : paramGrid[1][1]) {
            //         for (toMutate : paramGrid[1][2]) {
                        // starttime = System.currentTimeMillis();
                        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int) popSize, (int) (popSize * toMate),
                                (int) (popSize * toMutate), gap);
                        fit = new FixedIterationTrainer(ga, (int) iter);
                        fit.train();
                        System.out.println("\n" + iter + "iter, GA || popSize: " + popSize + ", toMate: " + toMate + ", toMutate: "
                                + toMutate + "\n\t" + ef.value(ga.getOptimal()));
                        System.out.println(ef.foundConflict());
                        // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            //         }
            //     }
            // }
        }

        System.out.println("============================");
        double samples = 200, toKeep = 50;
        for (double iter : paramGrid[3][1]) {
            // MIMIC grid search
            // for (double samples : paramGrid[2][0]) {
            //     for (double toKeep : paramGrid[2][1]) {
                    // starttime = System.currentTimeMillis();
                    MIMIC mimic = new MIMIC((int) samples, (int) toKeep, pop);
                    fit = new FixedIterationTrainer(mimic, (int) iter);
                    fit.train();
                    System.out.println("\n" + iter + "iter, MIMIC || samples: " + samples + ", toKeep: " + toKeep + "\n\t"
                            + ef.value(mimic.getOptimal()));
                    System.out.println(ef.foundConflict());
                    // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            //     }
            // }
        }


    }
}
