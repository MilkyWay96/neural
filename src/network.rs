use nalgebra::{DVector};
use rand::{distr::Distribution};
use thiserror::Error;

use crate::{
    activations::ActivationFn,
    dataset::Sample,
    losses::{self, LossFn},
};

use layer::{Layer, LayerError};

pub mod layer;

pub struct Network {
    layers: Vec<Layer>,
}

pub struct NetworkCache {
    activations: Vec<DVector<f32>>,
    weighted_inputs: Vec<DVector<f32>>,
}

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("too few layers ({0}) were specified in the constructor, at least two (input layer and output layer) are needed")]
    TooFewLayers(usize),

    #[error("layer {0}'s size has to be more than 0")]
    ZeroLayerSize(usize),

    #[error("{0}")]
    LayerError(#[from] LayerError),

    #[error("{0}")]
    LossFnError(#[from] losses::LossFnError),
}

fn check_layer_sizes(layer_sizes: &[usize]) -> Result<(), NetworkError> {
    if layer_sizes.len() < 2 {
        return Err(NetworkError::TooFewLayers(layer_sizes.len()));
    }
    
    if let Some(layer_index) = layer_sizes.iter().position(|&x| x == 0) {
        return Err(NetworkError::ZeroLayerSize(layer_index));
    }

    Ok(())
}

fn construct_layers<F>(layer_sizes: &[usize], constructor: F) -> Result<Vec<Layer>, LayerError> 
where
    F: Fn(usize, usize) -> Result<Layer, LayerError>
{
    layer_sizes
        .iter()
        .zip(layer_sizes.iter().skip(1))
        .map(|(&input_size, &output_size)| constructor(input_size, output_size))
        .collect()
}

impl Network {
    pub fn zeros(layer_sizes: &[usize], activation_fn: Box<dyn ActivationFn>) -> Result<Self, NetworkError> {
        check_layer_sizes(layer_sizes)?;

        let layers: Vec<Layer> = construct_layers(layer_sizes, |input_size, output_size| Layer::zeros(
            input_size,
            output_size,
            activation_fn.clone(),
        ))?;

        Ok(Self { layers })
    }

    pub fn random(
        layer_sizes: &[usize],
        activation_fn: Box<dyn ActivationFn>,
        distribution: &impl Distribution<f32>
    ) -> Result<Self, NetworkError> {
        check_layer_sizes(layer_sizes)?;

        let layers: Vec<Layer> = construct_layers(layer_sizes, |input_size, output_size| Layer::random(
            input_size,
            output_size,
            activation_fn.clone(),
            distribution,
        ))?;

        Ok(Self { layers })
    }

    pub fn forward(&mut self, input: DVector<f32>) -> Result<DVector<f32>, NetworkError> {
        self.layers.iter_mut().try_fold(input, |activations, layer| {
            layer.forward(activations).map_err(Into::into)
        })
    }

    pub fn backpropagate(&mut self, dataset: &[Sample], loss: &impl LossFn) -> Result<(), NetworkError> {
        for sample in dataset.iter() {
            let outputs = self.forward(sample.inputs().into_owned())?;
            let mut activation_partial_gradient = loss.partial_gradient(outputs.as_view(), sample.expected_outputs())?;

            activation_partial_gradient = self.layers.last_mut().unwrap().backpropagation_step(
                outputs.as_view(),
                activation_partial_gradient.as_view()
            );

            for i in (0..self.layers.len() - 1).rev() {
                let (left, right) = self.layers.split_at_mut(i + 1);
                let previous_input = right[0].get_previous_input();
                activation_partial_gradient = left[i]
                    .backpropagation_step(previous_input, activation_partial_gradient.as_view());
            }
        }

        Ok(())
    }

    pub fn learn(&mut self, dataset: &[Sample], loss: &impl LossFn, rate: f32) -> Result<(), NetworkError> {
        if dataset.is_empty() {
            return Ok(());
        }

        self.backpropagate(dataset, loss)?;

        for layer in self.layers.iter_mut() {
            layer.apply_gradient(-rate / dataset.len() as f32);
        }

        Ok(())
    }
}