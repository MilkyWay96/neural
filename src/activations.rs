pub trait ActivationFn: 'static + ActivationFnClone {
    fn apply(&self, x: f32) -> f32;
    fn derivative(&self, x: f32, activation: f32) -> f32;
}

pub trait ActivationFnClone {
    fn clone_box(&self) -> Box<dyn ActivationFn>;
}

impl<T> ActivationFnClone for T
where
    T: 'static + ActivationFn + Clone,
{
    fn clone_box(&self) -> Box<dyn ActivationFn> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ActivationFn> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone)]
pub struct Sigmoid;
impl ActivationFn for Sigmoid {
    fn apply(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f32, activation: f32) -> f32 {
        activation * (1.0 - activation)
    }
}

#[macro_export]
macro_rules! sigmoid {
    () => {
        Box::new(Sigmoid)
    };
}

pub use sigmoid;