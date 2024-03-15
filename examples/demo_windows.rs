mod common;

use common::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "demo windows";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    let mut demo = egui_demo_lib::DemoWindows::default();
    App::new(APP_NAME)?.run(
        move |_, ctx| {
            demo.ui(ctx);
        },
        move |_| {},
    )?;

    Ok(())
}
