// This is just an interactive test of the library

use macroquad::prelude::*;
use ::rand::distr::Uniform;
use nalgebra::dvector;

use neural::network::*;
use neural::activations::*;
use neural::losses;
use neural::dataset::Sample;

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
#[allow(unused_variables)]
async fn main() {
    let mut buffer = [BLACK; BUFFER_ROWS * BUFFER_COLUMNS];

    let mut network = Network::random(&[2, 50, 1], sigmoid!(), &Uniform::new(-0.5, 0.5).unwrap()).unwrap();

    let mut dataset = Vec::<Sample>::new();

    loop {
        let (mut mx, mut my) = mouse_position();
        mx = mx / screen_width() * 2.0 - 1.0;
        my = my / screen_height() * -2.0 + 1.0;

        if is_mouse_button_pressed(MouseButton::Left) {
            dataset.push(Sample::new(
                dvector![mx, my],
                dvector![1.0],
            ))
        }

        if is_mouse_button_pressed(MouseButton::Right) {
            dataset.push(Sample::new(
                dvector![mx, my],
                dvector![0.0],
            ))
        }

        for _ in 0..1000 {
            network.learn(&dataset, &losses::MSE, 0.01).unwrap();
        }

        for row in 0..BUFFER_ROWS {
            for column in 0..BUFFER_COLUMNS {
                let Some(color) = buffer.get_mut(column + row * BUFFER_COLUMNS) else { continue; };

                let output = network.forward(dvector![
                    column as f32 / BUFFER_COLUMNS as f32 * 2.0 - 1.0,
                    row as f32 / BUFFER_ROWS as f32 * -2.0 + 1.0,
                ]).unwrap()[0];

                *color = Color::new(output, 0.3, 1.0 - output, 1.0);
            }
        }

        draw_buffer(&mut buffer, BUFFER_ROWS, BUFFER_COLUMNS);

        for point in dataset.iter() {
            let pos = point.inputs();
            draw_circle(
                (pos[0] + 1.0) / 2.0 * screen_width(),
                (pos[1] - 1.0) / -2.0 * screen_height(),
                5.0,
                if point.expected_outputs()[0] == 1.0 { RED } else { BLUE },
            );
        }

        next_frame().await;
    }
}
