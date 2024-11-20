use std::{cell::RefCell, rc::Rc};

use anyhow::{anyhow, Result};
use gtk::{
    gdk::{Display, EventMask, EventType},
    gdk_pixbuf::{InterpType, Pixbuf},
    glib::Propagation,
    prelude::*,
    Application, ApplicationWindow, Box as GtkBox, Button, Image, Label, Orientation,
};
use tracing::{debug, error, info, span, trace, warn, Level};

pub struct WorldToCameraCalibrator {
    window: ApplicationWindow,
    points: Rc<RefCell<Vec<(f64, f64)>>>,
    selecting_points: Rc<RefCell<bool>>,
    points_label_zero: Rc<RefCell<Option<[(f64, f64); 4]>>>,
    points_silver_mineral: Rc<RefCell<Option<[(f64, f64); 3]>>>,
    point_outpose_guiding_light: Rc<RefCell<Option<(f64, f64)>>>,
    point_base_guiding_light: Rc<RefCell<Option<(f64, f64)>>>,
    point_power_rune_center: Rc<RefCell<Option<(f64, f64)>>>,
}

impl WorldToCameraCalibrator {
    pub fn new(app: &Application, image_path: &str) -> Result<Self> {
        let span = span!(Level::TRACE, "WorldToCameraCalibrator::new");
        let _enter = span.enter();

        let (scaled_width, scaled_height) = match Self::get_fit_window_size(image_path) {
            Ok((width, height)) => {
                info!("Window size is set to {}x{}", width, height);
                (width, height)
            }
            Err(e) => {
                error!("Failed to get fit window size for '{}': {}", image_path, e);
                warn!("Using default size 1296x1024");
                (1296, 1024)
            }
        };
        let window = ApplicationWindow::builder()
            .application(app)
            .title("World To Camera Calibrator")
            .default_width(scaled_width)
            .default_height(scaled_height)
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

        let points = Rc::new(RefCell::new(Vec::new()));
        let selecting_points = Rc::new(RefCell::new(false));

        let calibrator = Self {
            window,
            points,
            selecting_points,
            points_label_zero: Rc::new(RefCell::new(None)),
            points_silver_mineral: Rc::new(RefCell::new(None)),
            point_outpose_guiding_light: Rc::new(RefCell::new(None)),
            point_base_guiding_light: Rc::new(RefCell::new(None)),
            point_power_rune_center: Rc::new(RefCell::new(None)),
        };

        let image_pixbuf = Rc::new(RefCell::new(image_pixbuf));
        let image_widget = Rc::new(RefCell::new(image_widget));
        calibrator.connect_point_selection(image_widget.clone());
        calibrator.connect_size_allocate(image_pixbuf.clone(), image_widget.clone());

        Ok(calibrator)
    }

    pub fn show(&self) {
        self.window.show_all();
    }

    fn get_fit_window_size(image_path: &str) -> Result<(i32, i32)> {
        let span = span!(Level::TRACE, "WorldToCameraCalibrator::get_fit_window_size");
        let _enter = span.enter();

        let monitor_geometry = Display::default()
            .ok_or_else(|| anyhow!("Failed to get default display"))?
            .primary_monitor()
            .ok_or_else(|| anyhow!("Failed to get primary monitor"))?
            .geometry();
        let screen_width = monitor_geometry.width();
        let screen_height = monitor_geometry.height();

        debug!("Screen size: {screen_width}x{screen_height}");

        let image_pixbuf = Pixbuf::from_file(image_path)?;
        let image_width = image_pixbuf.width() as f32;
        let image_height = image_pixbuf.height() as f32;
        debug!("Image size: {image_width}x{image_height}");

        let aspect_ratio = image_width / image_height;
        let max_width = screen_width as f32 * 0.8;
        let max_height = screen_height as f32 * 0.8;
        let mut scaled_width = max_width;
        let mut scaled_height = scaled_width / aspect_ratio;
        if scaled_height > max_height {
            scaled_height = max_height;
            scaled_width = scaled_height * aspect_ratio;
        }

        debug!("Scaled width gotten: {scaled_width}x{scaled_height}");
        Ok((scaled_width as i32, scaled_height as i32))
    }

    fn connect_size_allocate(
        &self,
        image_pixbuf: Rc<RefCell<Pixbuf>>,
        image_widget: Rc<RefCell<Image>>,
    ) {
        let span = span!(
            Level::TRACE,
            "WorldToCameraCalibrator::connect_size_allocate"
        );
        let _enter = span.enter();

        self.window.connect_size_allocate({
            move |win, _| {
                trace!("Size allocate event triggered.");

                let width = win.default_width();
                let height = win.default_height();
                if let Some(scaled_pixbuf) =
                    image_pixbuf
                        .borrow_mut()
                        .scale_simple(width, height, InterpType::Bilinear)
                {
                    trace!("Successfully set image pixbuf size to {width}x{height}");
                    image_widget
                        .borrow_mut()
                        .set_from_pixbuf(Some(&scaled_pixbuf));
                } else {
                    error!("Failed to scale image to {width}x{height}");
                }
            }
        });
        trace!("Window connected to size allocate event;");
    }

    fn connect_point_selection(&self, image_widget: Rc<RefCell<Image>>) {
        let span = span!(
            Level::TRACE,
            "WorldToCameraCalibrator::connect_point_selection"
        );
        let _enter = span.enter();

        let points = self.points.clone();
        let selecting_points = self.selecting_points.clone();

        image_widget
            .borrow()
            .add_events(EventMask::BUTTON_PRESS_MASK | EventMask::BUTTON_RELEASE_MASK);

        self.window.connect_button_press_event(move |_, event| {
            trace!("Button press event triggered.");

            let (x, y) = event.position();
            let event_button = event.button();
            let event_type = event.event_type();
            if event_button == 1 {
                match event_type {
                    EventType::DoubleButtonPress => {
                        debug!("Left button with double press detected.");
                        if *selecting_points.borrow() {
                            *selecting_points.borrow_mut() = false;
                            points.borrow_mut().pop();
                            info!("Finished selecting points. Points: {:?}", points.borrow());
                            Propagation::Stop
                        } else {
                            *selecting_points.borrow_mut() = true;
                            points.borrow_mut().clear();
                            points.borrow_mut().push((x, y));
                            info!("Started selecting points. First point: ({}, {})", x, y);
                            Propagation::Stop
                        }
                    }
                    EventType::ButtonPress => {
                        debug!("Left button with single press detected.");
                        if *selecting_points.borrow() {
                            points.borrow_mut().push((x, y));
                            info!("Added point: ({}, {})", x, y);
                            Propagation::Stop
                        } else {
                            trace!("Ignored because points selection is not started.");
                            Propagation::Proceed
                        }
                    }
                    _ => {
                        trace!(
                            "Left button with event {} detected and ignored.",
                            event_type
                        );
                        Propagation::Proceed
                    }
                }
            } else if event_button == 3 && event_type == EventType::ButtonPress {
                debug!("Right button with single press detected.");
                if *selecting_points.borrow() {
                    *selecting_points.borrow_mut() = false;
                    points.borrow_mut().clear();
                    info!("Cancelled selecting points.");
                    Propagation::Stop
                } else {
                    trace!("Ignored because point selection not started.");
                    Propagation::Proceed
                }
            } else {
                debug!(
                    "Button {} with event {} detected and ignored.",
                    event_button, event_type
                );
                Propagation::Proceed
            }
        });
        trace!("Window connected to point selection event.");
    }
}
