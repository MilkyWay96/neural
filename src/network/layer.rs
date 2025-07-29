use nalgebra::{
    DMatrix,
    DVector,
    DVectorView,
};

use rand::{
    distr::{Distribution},
    Rng,
};

use thiserror::Error;

use crate::activations::ActivationFn;

pub struct Layer {
    weights: DMatrix<f32>,
    weight_gradient: DMatrix<f32>,
    biases: DVector<f32>,
    bias_gradient: DVector<f32>,
    activation_fn: Box<dyn ActivationFn>,

    previous_inputs: DVector<f32>,
    previous_weighted_sums: DVector<f32>,
}

#[derive(Debug, Error)]
pub enum LayerError {
    #[error("this layer takes {layer_input_size} inputs, but {given_input_size} were given")]
    InputSizeMismatch {
        layer_input_size: usize,
        given_input_size: usize,
    },

    #[error("input size has to be more than 0")]
    ZeroInputSize,

    #[error("output size has to be more than 0")]
    ZeroOutputSize,
}

fn check_sizes(input_size: usize, output_size: usize) -> Result<(), LayerError> {
    if input_size == 0 {
        return Err(LayerError::ZeroInputSize);
    }

    if output_size == 0 {
        return Err(LayerError::ZeroOutputSize);
    }

    Ok(())
}

fn random_vec<T>(size: usize, distribution: &impl Distribution<T>) -> Vec<T> {
    let rng = rand::rng();
    rng.sample_iter(distribution).take(size).collect()
}

impl Layer {
    pub fn zeros(
        input_size: usize,
        output_size: usize,
        activation_fn: Box<dyn ActivationFn>,
    ) -> Result<Self, LayerError> {
        check_sizes(input_size, output_size)?;

        Ok(Self {
            weights: DMatrix::zeros(output_size, input_size),
            weight_gradient: DMatrix::zeros(output_size, input_size),
            biases: DVector::zeros(output_size),
            bias_gradient: DVector::zeros(output_size),
            activation_fn,

            previous_inputs: DVector::zeros(input_size),
            previous_weighted_sums: DVector::zeros(output_size),
        })
    }

    pub fn random(
        input_size: usize,
        output_size: usize,
        activation_fn: Box<dyn ActivationFn>,
        distribution: &impl Distribution<f32>,
    ) -> Result<Self, LayerError> {
        check_sizes(input_size, output_size)?;

        Ok(Self {
            weights: DMatrix::from_vec(
                output_size,
                input_size,
                random_vec(output_size * input_size, distribution),
            ),

            weight_gradient: DMatrix::zeros(output_size, input_size),

            biases: DVector::from_vec(
                random_vec(output_size, distribution)
            ),

            bias_gradient: DVector::zeros(output_size),

            activation_fn,

            previous_inputs: DVector::zeros(input_size),
            previous_weighted_sums: DVector::zeros(output_size),
        })
    }

    pub fn forward(&mut self, inputs: DVector<f32>) -> Result<DVector<f32>, LayerError> {
        self.check_input_size(inputs.len())?;
        self.previous_weighted_sums = &self.weights * &inputs + &self.biases;
        self.previous_inputs = inputs;
        Ok(self.previous_weighted_sums.map(|x| self.activation_fn.apply(x)))
    }

    pub fn backpropagation_step(&mut self, previous_outputs: DVectorView<f32>, output_partial_gradient: DVectorView<f32>) -> DVector<f32> {
        let mut input_partial_gradient = DVector::zeros(self.input_size());

        for output_index in 0..self.output_size() {
            let activation_function_derivative = self.activation_fn.derivative(
                self.previous_weighted_sums[output_index],
                previous_outputs[output_index],
            );

            let bias_partial_derivative = activation_function_derivative * output_partial_gradient[output_index];
            self.bias_gradient[output_index] += bias_partial_derivative;

            for input_index in 0..self.input_size() {
                self.weight_gradient[(output_index, input_index)] += self.previous_inputs[input_index] * bias_partial_derivative;
                input_partial_gradient[input_index] += self.weights[(output_index, input_index)] * bias_partial_derivative;
            }
        }

        input_partial_gradient
    }

    pub fn apply_gradient(&mut self, scale: f32) {
        self.weights += &self.weight_gradient * scale;
        self.biases += &self.bias_gradient * scale;
        self.weight_gradient.fill(0.0);
        self.bias_gradient.fill(0.0);
    }

    #[inline]
    pub fn input_size(&self) -> usize { self.weights.ncols() }

    #[inline]
    pub fn output_size(&self) -> usize { self.weights.nrows() }

    #[inline]
    pub fn get_weight(&self, input: usize, output: usize) -> Option<&f32> {
        self.weights.get((output, input))
    }

    #[inline]
    pub fn get_weight_mut(&mut self, input: usize, output: usize) -> Option<&mut f32> {
        self.weights.get_mut((output, input))
    }

    #[inline]
    pub fn get_bias(&self, output: usize) -> Option<&f32> {
        self.biases.get(output)
    }

    #[inline]
    pub fn get_bias_mut(&mut self, output: usize) -> Option<&mut f32> {
        self.biases.get_mut(output)
    }

    pub fn get_previous_input(&self) -> DVectorView<f32> {
        self.previous_inputs.as_view()
    }

    fn check_input_size(&self, input_size: usize) -> Result<(), LayerError> {
        if self.input_size() != input_size {
            return Err(LayerError::InputSizeMismatch {
                layer_input_size: self.input_size(),
                given_input_size: input_size,
            });
        }

        Ok(())
    }
}