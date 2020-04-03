package NeuralNet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Model {
    ArrayList<Layer> layers;
    ArrayList<double[][]> weights;
    ArrayList<double[][]> biases;
    double learning_rate;
    long seed;

    public Model(){
        layers = new ArrayList<>();
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        seed = 50;
    }

    public Model(long seed){
        layers = new ArrayList<>();
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        this.seed = seed;
    }

    public void Add(Layer layer){
        layers.add(layer);
    }

    private void intializeModel(){
        double[][] layer_weights;
        double[][] layer_biases;

        for(int i=1; i<layers.size(); i++){
            int rows = layers.get(i-1).nodes;
            int columns = layers.get(i).nodes;

            layer_weights = new double[rows][columns];
            layer_biases = new double[1][columns];
            intializeWeights(layer_weights);
            weights.add(layer_weights);
            intializeBias(layer_biases);
            biases.add(layer_biases);
        }


    }

    private void intializeWeights(double[][] layer_weights){
        Random random = new Random();
        random.setSeed(seed);
        for(int i=0; i<layer_weights.length; i++){
            for(int j=0; j<layer_weights[0].length; j++){
                layer_weights[i][j] = -1.0 + (random.nextDouble()*(1 - (-1)));//generate random numbers between [-1, 1]
            }
        }
    }

    private void intializeBias(double[][] layer_bias){
        Random random = new Random();
        random.setSeed(seed);
        for(int i=0; i<layer_bias.length; i++){
            for(int j=0; j<layer_bias[0].length; j++){
                layer_bias[i][j] = -1.0 + (random.nextDouble()*(1 - (-1)));//generate random numbers between [-1, 1]
            }
        }
    }

    private double[][] updateWeights(double[][] old_wieghts, double[][] delta_weights, double learning_rate){
        double[][] new_weights = new double[old_wieghts.length][old_wieghts[0].length];
        for(int i=0; i<old_wieghts.length; i++){
            for(int j=0; j<old_wieghts[0].length; j++){
                new_weights[i][j] = (old_wieghts[i][j] - (delta_weights[i][j] * learning_rate));
            }
        }


        return new_weights;
    }

    private double[][] updateBias(double[][] old_bias, double[][] delta_bias, double learning_rate){
        double[][] new_biases = new double[old_bias.length][old_bias[0].length];
        for(int i=0; i<old_bias.length; i++){
            for(int j=0; j<old_bias[0].length; j++){
                new_biases[i][j] = (old_bias[i][j] - (delta_bias[i][j] * learning_rate));
            }
        }


        return new_biases;
    }



    private void forwardPass(){
        Matrix matrix = new Matrix();
        for(int i=1; i<layers.size(); i++){
            layers.get(i).set_activation(matrix.Multiplication(layers.get(i-1).get_activation(), weights.get(i-1)));
            layers.get(i).set_activation(matrix.correspondAddition(layers.get(i).get_activation(), biases.get(i-1)));
            if(i == layers.size()-1){
                layers.get(i).applyActivation("softmax");
            }
            else{
                layers.get(i).applyActivation();
            }
        }


        //System.out.println("We are here!! "+ layers.get(2).getSum());
    }

    private void backwardPass(double[][] X, double[][] Y, double learning_rate){
        //This is where the weights are distributed backward...
        Matrix matrix = new Matrix();
        double[][] delta_Y = new double[Y.length][Y[0].length];
        double[][] output = layers.get(layers.size()-1).z;
        for(int i=0; i<Y[0].length; i++){
            delta_Y[0][i] = output[0][i] - Y[0][i];
        }


        double[][] current_delta_z = null;
        double[][] prev_delta_z = delta_Y;
        double[][] temp_weights = weights.get(weights.size()-1);

        double[][] delta_w = matrix.Multiplication(matrix.Transpose(layers.get(layers.size()-2).z), delta_Y);

        weights.set(weights.size()-1, updateWeights(weights.get(weights.size()-1), delta_w, learning_rate));
        biases.set(biases.size()-1, updateBias(biases.get(biases.size()-1), delta_Y, learning_rate));

        for(int i=layers.size()-2; i>0; i--){
            current_delta_z = matrix.Multiplication(prev_delta_z, matrix.Transpose(temp_weights));
            current_delta_z = matrix.correspondMultiplication(current_delta_z, layers.get(i).sigmoidDerivative());

            temp_weights = weights.get(i-1);

            delta_w = matrix.Multiplication(matrix.Transpose(layers.get(i-1).z), current_delta_z);
            weights.set(i-1, updateWeights(weights.get(i-1), delta_w, learning_rate));
            biases.set(i-1, updateBias(biases.get(i-1), current_delta_z, learning_rate));
            prev_delta_z = current_delta_z;

        }

    }

    private double calculateError(double[][][] X, double[][][] Y){
        //Note: This will calculate error according to the formula of categorical cross entropy..
        Matrix matrix  = new Matrix();
        double error_sum = 0.0;

        for(int i=0; i<X.length; i++){
            layers.get(0).feed_input(X[i]);
            forwardPass();
            double[][] temp_Y = matrix.scalarMultiply(matrix.copyMatrix(Y[i]), -1);
            double[][] log_activation = matrix.copyMatrix(layers.get(layers.size()-1).z);
            log_activation = matrix.logMatrix(log_activation);
            double[][] errorMatrix = matrix.correspondMultiplication(temp_Y, log_activation);
            double error = matrix.sumMatrix(errorMatrix);
            error_sum+=error;
        }

        return error_sum;
    }



    public void Train(double[][][] X, double[][][] Y, double learning_rate, int epochs){


        intializeModel();

        System.out.println("epoch: "+0 +" loss: "+calculateError(X, Y));
        for(int e=1; e<=epochs; e++){


            for(int i=0; i<X.length; i++){

                layers.get(0).feed_input(X[i]);
                forwardPass();
                backwardPass(X[i], Y[i], learning_rate);


            }

            if(e%500 == 0 || e == epochs)
            System.out.println("epoch: "+e +" loss: "+calculateError(X, Y));

        }

    }

    public void Test(double[][] X, double[][] Y, int count){
        for(int i=0; i<X.length; i++){
            layers.get(0).feed_input(X);
            forwardPass();

            System.out.println("Item No: "+ count);

            double c1 = layers.get(layers.size()-1).z[0][0];
            double c2 = layers.get(layers.size()-1).z[0][1];
            double c3 = layers.get(layers.size()-1).z[0][2];

            ArrayList<Double> softmaxValuesList = new ArrayList<>();


            for(int j=0; j<layers.get(layers.size()-1).z[0].length; j++){
                softmaxValuesList.add(layers.get(layers.size()-1).z[0][j]);
            }

            double pred_max = Collections.max(softmaxValuesList);
            int pred_max_index = softmaxValuesList.indexOf(pred_max)+1;
            System.out.println("Predicted Class: "+ pred_max_index);


        }
    }


}
