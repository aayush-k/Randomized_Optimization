package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Modified to ContinuousPeaksTest
 * @version 1.0
 */
public class AKContinuousPeaksTest {
    /** The n value */
    private static final int N = 500;
    /** The t value */
    private static final int T = N / 2;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
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
                { 1, 5, 10, 25, 500, 1000, 50000} // ga, mimic
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
            // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        }
        System.out.println("============================");
        double t0 = 2500, coolingRate = 0.25;

        for (double iter : paramGrid[3][0]) {

            // // SA grid search
            // for (t0 : paramGrid[0][0]) {
            //     for (coolingRate : paramGrid[0][1]) {
            // starttime = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(t0, coolingRate, hcp);
            fit = new FixedIterationTrainer(sa, (int) iter);
            fit.train();
            System.out.println("\n" + iter + " iter, SA || t0: " + t0 + ", coolingRate: " + coolingRate + "\n\t"
                    + ef.value(sa.getOptimal()));
            // System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            //     }
            // }
        }


        System.out.println("============================");
        double popSize = 250, toMate = 0.05, toMutate = 0.03;

        for (double iter : paramGrid[3][1]) {
            // GA grid search
            // for (double popSize : paramGrid[1][0]) {
            //     for (double toMate : paramGrid[1][1]) {
            //         for (double toMutate: paramGrid[1][2]) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int)popSize, (int) (popSize * toMate),
                    (int) (popSize * toMutate), gap);
            fit = new FixedIterationTrainer(ga, (int) iter);
            fit.train();
            System.out.println("\n" + iter + " iter, GA || popSize: " + popSize + ", toMate: " + toMate + ", toMutate: "
                    + toMutate + "\n\t" + ef.value(ga.getOptimal()));
            //         }
            //     }
            // }
        }

        // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        // System.out.println("GA: " + ef.value(ga.getOptimal()));

        System.out.println("============================");
        double samples = 150, toKeep = 10;
        for (double iter : paramGrid[3][1]) {
            // MIMIC grid search
            // for (samples: paramGrid[2][0]) {
            //     for (toKeep: paramGrid[2][1]) {
            MIMIC mimic = new MIMIC((int) samples, (int) toKeep, pop);
            fit = new FixedIterationTrainer(mimic, (int) iter);
            fit.train();
            System.out.println("\n" + iter + " iter, MIMIC || samples: " + samples + ", toKeep: " + toKeep + "\n\t"
                    + ef.value(mimic.getOptimal()));
            //     }
            // }
        }

        // MIMIC mimic = new MIMIC(200, 20, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        // System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    }
}
