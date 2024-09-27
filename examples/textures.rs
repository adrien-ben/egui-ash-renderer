mod common;

use ash::vk;
use common::{run_app, vulkan::texture::Texture, App, System, VulkanContext};
use egui::{load::SizedTexture, Vec2, Widget};
use egui_ash_renderer::vulkan::*;
use simple_logger::SimpleLogger;
use std::error::Error;

struct TexturesDemo {
    srgb_texture: UserTexture,
    linear_texture: UserTexture,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    show_srgb_texture: bool,
}

impl App for TexturesDemo {
    fn title() -> &'static str {
        "textures"
    }

    fn new(app: &mut System) -> Self {
        let VulkanContext {
            physical_device,
            device,
            ..
        } = &app.vulkan_context;

        let memory_properties = unsafe {
            app.vulkan_context
                .instance
                .get_physical_device_memory_properties(*physical_device)
        };
        let descriptor_set_layout = create_vulkan_descriptor_set_layout(device).unwrap();
        let descriptor_pool = create_vulkan_descriptor_pool(device, 2).unwrap();

        let srgb_texture = UserTexture::from_memory(
            app,
            memory_properties,
            descriptor_set_layout,
            descriptor_pool,
            vk::Format::R8G8B8A8_SRGB,
            include_bytes!("../assets/images/img2.jpg"),
        );

        let linear_texture = UserTexture::from_memory(
            app,
            memory_properties,
            descriptor_set_layout,
            descriptor_pool,
            vk::Format::R8G8B8A8_UNORM,
            include_bytes!("../assets/images/normals.jpg"),
        );

        let show_srgb_texture = true;

        Self {
            srgb_texture,
            linear_texture,
            descriptor_set_layout,
            descriptor_pool,

            show_srgb_texture,
        }
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Managed texture")
            .show(ctx, |ui| {
                ui.label("This texture is loaded and managed by egui. Loaders must be installed for it to work.");
                egui::Image::new(egui::include_image!("../assets/images/img1.jpg")).fit_to_original_size(0.8).ui(ui);
            });

        egui::Window::new("Used defined texture").show(ctx, |ui| {
            ui.label("This texture is loaded and managed by the user.");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.show_srgb_texture, true, "sRGB");
                ui.radio_value(&mut self.show_srgb_texture, false, "Linear");
            });

            let texture = if self.show_srgb_texture {
                self.srgb_texture.egui_texture
            } else {
                self.linear_texture.egui_texture
            };
            egui::Image::new(texture).fit_to_original_size(0.8).ui(ui);
        });
    }

    fn clean(&mut self, vulkan_ctx: &VulkanContext) {
        let device = &vulkan_ctx.device;
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
        self.srgb_texture.texture.destroy(device);
        self.linear_texture.texture.destroy(device);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    run_app::<TexturesDemo>()?;
    Ok(())
}

struct UserTexture {
    texture: Texture,
    _set: vk::DescriptorSet,
    egui_texture: SizedTexture,
}

impl UserTexture {
    fn from_memory(
        app: &mut System,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        set_layout: vk::DescriptorSetLayout,
        set_pool: vk::DescriptorPool,
        format: vk::Format,
        data: &[u8],
    ) -> Self {
        let image = image::load_from_memory(data).unwrap();
        let width = image.width();
        let height = image.height();
        let data = image.to_rgba8().into_vec();

        let texture = Texture::from_rgba8(
            &app.vulkan_context.device,
            app.vulkan_context.graphics_queue,
            app.vulkan_context.command_pool,
            memory_properties,
            width,
            height,
            format,
            data.as_slice(),
        )
        .unwrap();

        let set = create_vulkan_descriptor_set(
            &app.vulkan_context.device,
            set_layout,
            set_pool,
            texture.image_view,
            texture.sampler,
        )
        .unwrap();

        let texture_id = app.renderer.add_user_texture(set);

        let egui_texture = SizedTexture {
            id: texture_id,
            size: Vec2 {
                x: width as f32,
                y: height as f32,
            },
        };

        Self {
            texture,
            _set: set,
            egui_texture,
        }
    }
}
