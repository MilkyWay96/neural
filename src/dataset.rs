use nalgebra::{DVector, DVectorView};

pub struct Sample {
    inputs: DVector<f32>,
    expected_outputs: DVector<f32>,
}

impl Sample {
    pub fn new(inputs: DVector<f32>, expected_outputs: DVector<f32>) -> Self {
        Self {
            inputs,
            expected_outputs,
        }
    }

    pub fn inputs(&self) -> DVectorView<f32> {
        self.inputs.as_view()
    }

    pub fn expected_outputs(&self) -> DVectorView<f32> {
        self.expected_outputs.as_view()
    }
}