use nalgebra::DVectorView;
use thiserror::Error;

pub trait LossFn: LossFnChecked {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<f32, LossFnError>;

    fn partial_derivative(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
        with_respect_to: usize,
    ) -> Result<f32, LossFnError>;
}

#[derive(Debug, Error)]
pub enum LossFnError {
    #[error("given output size ({given_output_size}) does not equal expected output size ({expected_output_size})")]
    OutputSizeMismatch {
        given_output_size: usize,
        expected_output_size: usize,
    },

    #[error("output index ({output_index}) >= output size ({output_size})")]
    OutputIndexOutOfRange {
        output_size: usize,
        output_index: usize,
    },
}

pub trait LossFnChecked {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> f32;

    fn partial_derivative(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
        with_respect_to: usize,
    ) -> f32;
}

impl<T: LossFnChecked> LossFn for T {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> Result<f32, LossFnError> {
        if output.len() != expected_output.len() {
            return Err(LossFnError::OutputSizeMismatch {
                given_output_size: output.len(),
                expected_output_size: expected_output.len(),
            });
        }

        Ok(LossFnChecked::apply(self, output, expected_output))
    }

    fn partial_derivative(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
        with_respect_to: usize,
    ) -> Result<f32, LossFnError> {
        if output.len() != expected_output.len() {
            return Err(LossFnError::OutputSizeMismatch {
                given_output_size: output.len(),
                expected_output_size: expected_output.len(),
            });
        }

        if with_respect_to >= output.len() {
            return Err(LossFnError::OutputIndexOutOfRange {
                output_size: output.len(),
                output_index: with_respect_to,
            });
        }

        Ok(LossFnChecked::partial_derivative(self, output, expected_output, with_respect_to))
    }
}

pub struct MSE;
impl LossFnChecked for MSE {
    fn apply(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
    ) -> f32 {
        output
            .iter()
            .zip(expected_output.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>() / output.len() as f32
    }

    fn partial_derivative(
        &self,
        output: DVectorView<f32>,
        expected_output: DVectorView<f32>,
        with_respect_to: usize,
    ) -> f32 {
        2.0 * (output[with_respect_to] - expected_output[with_respect_to]) / output.len() as f32
    }
}