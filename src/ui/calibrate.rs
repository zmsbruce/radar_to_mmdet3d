use anyhow::{anyhow, Ok, Result};
use gtk::{
    gdk::Display,
    gdk_pixbuf::{InterpType, Pixbuf},
    prelude::*,
    Application, ApplicationWindow, Box as GtkBox, Button, Image, Label, Orientation,
};

pub struct WorldToCameraCalibrator {
    window: ApplicationWindow,
}

impl WorldToCameraCalibrator {
    pub fn new(app: &Application, image_path: &str) -> Result<Self> {
        let (scaled_width, scaled_height) =
            Self::get_fit_window_size(image_path).unwrap_or_else(|_| (1296, 1024));
        let window = ApplicationWindow::builder()
            .application(app)
            .title("World To Camera Calibrator")
            .default_width(scaled_width as i32)
            .default_height(scaled_height as i32)
            .resizable(false)
            .build();

        let image_pixbuf = Pixbuf::from_file(image_path)?;
        let image_widget = Image::from_pixbuf(Some(&image_pixbuf));

        let left_button = Button::with_label("Left");
        left_button.set_width_request(60);
        left_button.set_hexpand(false);
        left_button.set_margin_start(10);

        let right_button = Button::with_label("Right");
        right_button.set_hexpand(false);
        right_button.set_width_request(60);
        right_button.set_margin_end(10);

        let info_label = Label::new(Some("This is a label with some information."));
        info_label.set_halign(gtk::Align::Center);
        info_label.set_hexpand(true);

        let hbox = GtkBox::new(Orientation::Horizontal, 10);
        hbox.set_margin_top(10);
        hbox.add(&left_button);
        hbox.add(&info_label);
        hbox.add(&right_button);

        let vbox = GtkBox::new(Orientation::Vertical, 10);
        vbox.add(&hbox);
        vbox.add(&image_widget);

        window.set_child(Some(&vbox));

        let image_pixbuf_clone = image_pixbuf.clone();
        window.connect_size_allocate(move |win, _| {
            let width = win.default_width();
            let height = win.default_height();

            let scaled_pixbuf = image_pixbuf_clone
                .scale_simple(width, height, InterpType::Bilinear)
                .expect("Failed to scale image");

            image_widget.set_from_pixbuf(Some(&scaled_pixbuf));
        });

        Ok(Self { window })
    }

    pub fn show(&self) {
        self.window.show_all();
    }

    pub fn get_fit_window_size(image_path: &str) -> Result<(i32, i32)> {
        let monitor_geometry = Display::default()
            .ok_or_else(|| anyhow!("Failed to get default display"))?
            .primary_monitor()
            .ok_or_else(|| anyhow!("Failed to get primary monitor"))?
            .geometry();
        let screen_width = monitor_geometry.width();
        let screen_height = monitor_geometry.height();

        let image_pixbuf = Pixbuf::from_file(image_path)?;
        let image_width = image_pixbuf.width() as f32;
        let image_height = image_pixbuf.height() as f32;
        let aspect_ratio = image_width / image_height;

        let max_width = screen_width as f32 * 0.8;
        let max_height = screen_height as f32 * 0.8;
        let mut scaled_width = max_width;
        let mut scaled_height = scaled_width / aspect_ratio;
        if scaled_height > max_height {
            scaled_height = max_height;
            scaled_width = scaled_height * aspect_ratio;
        }

        Ok((scaled_width as i32, scaled_height as i32))
    }
}
