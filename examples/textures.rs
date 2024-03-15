mod common;

use common::{vulkan::texture::Texture, App, VulkanContext};
use egui::{load::SizedTexture, Vec2, Widget};
use egui_ash_renderer::vulkan::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "textures";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;

    let mut system = App::new(APP_NAME)?;

    let VulkanContext {
        physical_device,
        device,
        graphics_queue,
        command_pool,
        ..
    } = &system.vulkan_context;

    let image_bytes = include_bytes!("../assets/images/img2.jpg");
    let image = image::load_from_memory(image_bytes).unwrap();
    let width = image.width();
    let height = image.height();
    let data = image.to_rgba8().into_vec();

    let memory_properties = unsafe {
        system
            .vulkan_context
            .instance
            .get_physical_device_memory_properties(*physical_device)
    };

    let mut texture = Texture::from_rgba8(
        device,
        *graphics_queue,
        *command_pool,
        memory_properties,
        width,
        height,
        data.as_slice(),
    )
    .unwrap();

    let descriptor_set_layout = create_vulkan_descriptor_set_layout(device).unwrap();

    let descriptor_pool = create_vulkan_descriptor_pool(device, 1).unwrap();

    let descriptor_set = create_vulkan_descriptor_set(
        device,
        descriptor_set_layout,
        descriptor_pool,
        texture.image_view,
        texture.sampler,
    )
    .unwrap();

    let texture_id = system.renderer.add_user_texture(descriptor_set);

    let egui_texture = SizedTexture {
        id: texture_id,
        size: Vec2 {
            x: width as f32,
            y: height as f32,
        },
    };

    system.run( move |_, ctx| {
        egui::Window::new("Managed texture")
            .show(ctx, |ui| {
                ui.label("This texture is loaded and managed by egui. Loaders must be installed for it to work.");
                egui::Image::new(egui::include_image!("../assets/images/img1.jpg")).fit_to_original_size(0.8).ui(ui);
            });

        egui::Window::new("Used defined texture")
            .show(ctx, |ui| {
                ui.label("This texture is loaded and managed by the user.");
                egui::Image::new(egui_texture).fit_to_original_size(0.8).ui(ui);
            });
    },
    move |vulkan_ctx| {
        log::info!("Destroy texture app");
        unsafe {
            let device = &vulkan_ctx.device;
            device.destroy_descriptor_pool(descriptor_pool, None);
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            texture.destroy(device);
        }
    }
    )?;

    Ok(())
}
