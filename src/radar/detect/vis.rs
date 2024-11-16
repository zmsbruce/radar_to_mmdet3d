use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use once_cell::sync::Lazy;
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use rusttype::{point, Font, Scale};
use show_image::{
    event::{VirtualKeyCode, WindowEvent},
    AsImageView, WindowOptions,
};

use super::{BBox, RobotLabel};

static FONT_DATA: Lazy<&'static [u8]> =
    Lazy::new(|| include_bytes!("../../../assets/fonts/NotoSans-Regular.ttf") as &[u8]);

pub fn draw_rect_on_draw_target(dt: &mut DrawTarget, rect: &BBox, color: SolidSource, width: f32) {
    let mut pb = PathBuilder::new();
    pb.rect(
        rect.x_center - rect.width / 2.0,
        rect.y_center - rect.height / 2.0,
        rect.width,
        rect.height,
    );
    let path = pb.finish();

    dt.stroke(
        &path,
        &Source::Solid(color),
        &StrokeStyle {
            join: LineJoin::Round,
            width,
            ..StrokeStyle::default()
        },
        &DrawOptions::default(),
    );
}

pub fn get_color_from_robot_label(label: RobotLabel) -> SolidSource {
    match label {
        RobotLabel::BlueHero
        | RobotLabel::BlueEngineer
        | RobotLabel::BlueInfantryThree
        | RobotLabel::BlueInfantryFour
        | RobotLabel::BlueInfantryFive
        | RobotLabel::BlueSentry => SolidSource {
            r: 0,
            g: 0,
            b: 0xff,
            a: 0xff,
        },

        RobotLabel::RedHero
        | RobotLabel::RedEngineer
        | RobotLabel::RedInfantryThree
        | RobotLabel::RedInfantryFour
        | RobotLabel::RedInfantryFive
        | RobotLabel::RedSentry => SolidSource {
            r: 0xff,
            g: 0,
            b: 0,
            a: 0xff,
        },
    }
}

pub fn draw_text_on_draw_target(
    dt: &mut DrawTarget,
    text: &str,
    position: (f32, f32),
    font_size: f32,
) -> Result<()> {
    let font_data = *FONT_DATA;

    let font = Font::try_from_bytes(&font_data).context("Failed to load font")?;
    let font_scale = Scale::uniform(font_size);
    let v_metrics = font.v_metrics(font_scale);

    let (text_x, text_y) = position;
    let offset = point(text_x + font_size / 4.0, text_y + v_metrics.ascent);

    let glyphs: Vec<_> = font.layout(&text, font_scale, offset).collect();
    for glyph in glyphs {
        if let Some(bbox) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                dt.fill_rect(
                    (x as i32 + bbox.min.x) as f32,
                    (y as i32 + bbox.min.y) as f32,
                    1.0,
                    1.0,
                    &Source::Solid(SolidSource {
                        r: 0xff,
                        g: 0xff,
                        b: 0xff,
                        a: 0xff,
                    }),
                    &DrawOptions {
                        alpha: v,
                        ..DrawOptions::default()
                    },
                );
            });
        }
    }

    Ok(())
}

pub fn display_window_and_waitkey(image: &DynamicImage, draw_target: &DrawTarget) -> Result<()> {
    let (width, height) = image.dimensions();
    let image = image.clone();
    let overlay: show_image::Image = draw_target.into();

    let window = show_image::context().run_function_wait(move |context| -> Result<_> {
        let mut window = context
            .create_window(
                "vis",
                WindowOptions {
                    size: Some([width, height]),
                    ..WindowOptions::default()
                },
            )
            .context("Failed to create window")?;
        window.set_image(
            "picture",
            &image
                .as_image_view()
                .context("Failed to image view of original image")?,
        );
        window.set_overlay(
            "yolo",
            &overlay
                .as_image_view()
                .context("Failed to set image view of overlay")?,
            true,
        );
        Ok(window.proxy())
    })?;

    for event in window.event_channel().unwrap() {
        if let WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}
