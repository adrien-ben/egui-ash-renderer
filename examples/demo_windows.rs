mod common;

use common::*;
use egui_demo_lib::DemoWindows;
use simple_logger::SimpleLogger;
use std::error::Error;

impl App for DemoWindows {
    fn title() -> &'static str {
        "demo windows"
    }

    fn new(_: &mut System) -> Self {
        DemoWindows::default()
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        self.ui(ctx);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    run_app::<DemoWindows>()?;
    Ok(())
}
