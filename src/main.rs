// This is just an interactive test of the library

use macroquad::prelude::*;
use ::rand::distr::Uniform;
use nalgebra::dvector;

use neural::network::*;
use neural::activations::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Neural Network".to_string(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

fn draw_buffer(buffer: &[Color], rows: usize, columns: usize) {
    let width = screen_width() / columns as f32;
    let height = screen_height() / rows as f32;
    
    for row in 0..BUFFER_ROWS {
        for column in 0..BUFFER_COLUMNS {
            if let Some(&color) = buffer.get(column + row * BUFFER_COLUMNS) {
                draw_rectangle(column as f32 * width, row as f32 * height, width, height, color);
            }
        }
    }
}

const BUFFER_ROWS: usize = 120;
const BUFFER_COLUMNS: usize = 160;

#[macroquad::main(window_conf)]
async fn main() {
    let mut buffer = [BLACK; BUFFER_ROWS * BUFFER_COLUMNS];

    let network = Network::random(&[2, 10, 10, 1], sigmoid!(), &Uniform::new(-5.0, 5.0).unwrap()).unwrap();

    loop {
        for row in 0..BUFFER_ROWS {
            for column in 0..BUFFER_COLUMNS {
                let Some(color) = buffer.get_mut(column + row * BUFFER_COLUMNS) else { continue; };

                let output = network.forward(dvector![
                    column as f32 / BUFFER_COLUMNS as f32 * 2.0 - 1.0,
                    row as f32 / BUFFER_ROWS as f32 * 2.0 - 1.0,
                ].as_view()).unwrap()[0];

                *color = Color::new(output, 0.3, 1.0 - output, 1.0);
            }
        }

        draw_buffer(&mut buffer, BUFFER_ROWS, BUFFER_COLUMNS);

        next_frame().await;
    }
}
