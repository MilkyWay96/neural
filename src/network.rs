use nalgebra::{DVectorView, DVector};
use rand::distr::Distribution;
use thiserror::Error;

use crate::activations::ActivationFn;
use layer::{Layer, LayerError};

pub mod layer;

pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("too few layers ({0}) were specified in the constructor, at least two (input layer and output layer) are needed")]
    TooFewLayers(usize),

    #[error("layer {0}'s size has to be more than 0")]
    ZeroLayerSize(usize),

    #[error("{0}")]
    LayerError(#[from] LayerError),
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

    pub fn forward(&self, input: DVectorView<f32>) -> Result<DVector<f32>, NetworkError> {
        let activations = self.layers[0].forward(input)?;
        self.layers.iter().skip(1).try_fold(activations, |activations, layer| {
            layer.forward(activations.as_view()).map_err(Into::into)
        })
    }
}