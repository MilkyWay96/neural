use nalgebra::{
    DMatrix,
    DVector, DVectorView,
};

use rand::{
    distr::{Distribution},
    Rng,
};

use thiserror::Error;

use crate::activations::ActivationFn;

pub struct Layer {
    input_size: usize,
    output_size: usize,
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    activation_fn: Box<dyn ActivationFn>,
}

#[derive(Debug, Error)]
pub enum LayerError {
    #[error("this layer's input size is {layer_input_size}, but an input with size {given_input_size} was given")]
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
            input_size,
            output_size,
            weights: DMatrix::zeros(output_size, input_size),
            biases: DVector::zeros(output_size),
            activation_fn,
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
            input_size,
            output_size,

            weights: DMatrix::from_vec(
                output_size,
                input_size,
                random_vec(output_size * input_size, distribution),
            ),

            biases: DVector::from_vec(
                random_vec(output_size, distribution)
            ),

            activation_fn,
        })
    }

    pub fn forward(&self, input: DVectorView<f32>) -> Result<DVector<f32>, LayerError> {
        if self.input_size != input.len() {
            return Err(LayerError::InputSizeMismatch {
                layer_input_size: self.input_size,
                given_input_size: input.len(),
            });
        }

        let weighted_inputs = &self.weights * input + &self.biases;
        Ok(weighted_inputs.map(|x| self.activation_fn.apply(x)))
    }

    pub fn get_input_size(&self) -> usize { self.input_size }

    pub fn get_output_size(&self) -> usize { self.output_size }

    pub fn get_weight(&self, input: usize, output: usize) -> Option<&f32> {
        self.weights.get((output, input))
    }

    pub fn get_weight_mut(&mut self, input: usize, output: usize) -> Option<&mut f32> {
        self.weights.get_mut((output, input))
    }

    pub fn get_bias(&self, output: usize) -> Option<&f32> {
        self.biases.get(output)
    }

    pub fn get_bias_mut(&mut self, output: usize) -> Option<&mut f32> {
        self.biases.get_mut(output)
    }
}