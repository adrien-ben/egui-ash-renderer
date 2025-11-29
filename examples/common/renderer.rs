use ash::{Device, vk};
use egui::epaint::ImageDelta;
use egui::{ClippedPrimitive, TextureId};
use egui_ash_renderer::{Options, RenderMode, Renderer};
use egui_ash_renderer::{RendererResult, allocator::*};

use crate::common::{Swapchain, VulkanContext};

#[allow(dead_code)]
pub enum AnyRenderer {
    #[cfg(feature = "simple-allocator")]
    Simple(Renderer<SimpleAllocator>),
    #[cfg(feature = "gpu-allocator")]
    Gpu(Renderer<GpuAllocator>),
    #[cfg(feature = "vk-mem")]
    VkMem(Renderer<VkMemAllocator>),
    #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
    Custom(Renderer<TrackingAllocator<SimpleAllocator>>),
}

impl AnyRenderer {
    pub fn build(ctx: &VulkanContext, swapchain: &Swapchain) -> RendererResult<Self> {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))] {
                let memory_properties = unsafe {
                    ctx.instance.get_physical_device_memory_properties(ctx.physical_device)
                };
                let renderer = AnyRenderer::Custom(Renderer::with_custom_allocator(
                    TrackingAllocator::new(SimpleAllocator::new(memory_properties)),
                    ctx.device.clone(),
                    RenderMode::RenderPass(swapchain.render_pass),
                    Options {
                        srgb_framebuffer: true,
                        ..Default::default()
                    },
                )?);
            } else if #[cfg(feature = "simple-allocator")] {
                let renderer = AnyRenderer::Simple(Renderer::with_simple_allocator(
                    &ctx.instance,
                    ctx.physical_device,
                    ctx.device.clone(),
                    RenderMode::RenderPass(swapchain.render_pass),
                    Options {
                        srgb_framebuffer: true,
                        ..Default::default()
                    },
                )?);
            } else if #[cfg(feature = "gpu-allocator")] {
                let renderer = {
                    let allocator = gpu_allocator::vulkan::Allocator::new(
                        &gpu_allocator::vulkan::AllocatorCreateDesc {
                            instance: ctx.instance.clone(),
                            device: ctx.device.clone(),
                            physical_device: ctx.physical_device,
                            debug_settings: Default::default(),
                            buffer_device_address: false,
                            allocation_sizes: Default::default(),
                        },
                    )?;

                    AnyRenderer::Gpu(Renderer::with_gpu_allocator(
                        std::sync::Arc::new(std::sync::Mutex::new(allocator)),
                        ctx.device.clone(),
                        RenderMode::RenderPass(swapchain.render_pass),
                        Options {
                            srgb_framebuffer: true,
                            ..Default::default()
                        },
                    )?)
                };
            } else if #[cfg(feature = "vk-mem")] {
                let renderer = {
                    let allocator = {
                        let allocator_create_info = vk_mem::AllocatorCreateInfo::new(
                            &ctx.instance,
                            &ctx.device,
                            ctx.physical_device,
                        );

                        unsafe { vk_mem::Allocator::new(allocator_create_info)? }
                    };

                    AnyRenderer::VkMem(Renderer::with_vk_mem_allocator(
                        std::sync::Arc::new(allocator),
                        ctx.device.clone(),
                        RenderMode::RenderPass(swapchain.render_pass),
                        Options {
                            srgb_framebuffer: true,
                            ..Default::default()
                        },
                    )?)
                };
            }
        }

        log::info!("Selected {} renderer", renderer.get_name());
        Ok(renderer)
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(_) => "simple",
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(_) => "gpu",
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(_) => "vk-mem",
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(_) => "custom",
        }
    }

    pub fn set_render_mode(&mut self, render_mode: RenderMode) {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(r) => r.set_render_mode(render_mode),
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(r) => r.set_render_mode(render_mode),
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(r) => r.set_render_mode(render_mode),
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(r) => r.set_render_mode(render_mode),
        }
        .expect("Failed to rebuild renderer pipeline");
    }

    pub fn free_textures(&mut self, ids: &[TextureId]) {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(r) => r.free_textures(ids),
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(r) => r.free_textures(ids),
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(r) => r.free_textures(ids),
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(r) => r.free_textures(ids),
        }
        .expect("Failed to free textures");
    }

    pub fn set_textures(
        &mut self,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        textures_delta: &[(TextureId, ImageDelta)],
    ) {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(r) => r.set_textures(queue, command_pool, textures_delta),
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(r) => r.set_textures(queue, command_pool, textures_delta),
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(r) => r.set_textures(queue, command_pool, textures_delta),
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(r) => r.set_textures(queue, command_pool, textures_delta),
        }
        .expect("Failed to update texture")
    }

    #[allow(dead_code)]
    pub fn add_user_texture(&mut self, set: vk::DescriptorSet) -> TextureId {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(r) => r.add_user_texture(set),
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(r) => r.add_user_texture(set),
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(r) => r.add_user_texture(set),
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(r) => r.add_user_texture(set),
        }
    }

    pub fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        pixels_per_point: f32,
        primitives: &[ClippedPrimitive],
    ) -> RendererResult<()> {
        match self {
            #[cfg(feature = "simple-allocator")]
            AnyRenderer::Simple(r) => {
                r.cmd_draw(command_buffer, extent, pixels_per_point, primitives)
            }
            #[cfg(feature = "gpu-allocator")]
            AnyRenderer::Gpu(r) => r.cmd_draw(command_buffer, extent, pixels_per_point, primitives),
            #[cfg(feature = "vk-mem")]
            AnyRenderer::VkMem(r) => {
                r.cmd_draw(command_buffer, extent, pixels_per_point, primitives)
            }
            #[cfg(all(feature = "custom-allocator", feature = "simple-allocator"))]
            AnyRenderer::Custom(r) => {
                r.cmd_draw(command_buffer, extent, pixels_per_point, primitives)
            }
        }
    }
}

#[allow(dead_code)]
pub struct TrackingAllocator<A: Allocator> {
    delegate: A,
    buffer_created: u64,
    buffer_destroyed: u64,
    image_create: u64,
    image_destroyed: u64,
}

impl<A: Allocator> TrackingAllocator<A> {
    #[allow(dead_code)]
    fn new(delegate: A) -> Self {
        Self {
            delegate,
            buffer_created: 0,
            buffer_destroyed: 0,
            image_create: 0,
            image_destroyed: 0,
        }
    }
}

impl<A: Allocator> Allocator for TrackingAllocator<A> {
    type Allocation = A::Allocation;

    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Allocation)> {
        let res = self.delegate.create_buffer(device, size, usage)?;
        self.buffer_created += 1;
        Ok(res)
    }

    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &mut Self::Allocation,
        data: &[T],
    ) -> RendererResult<()> {
        self.delegate.update_buffer(device, memory, data)
    }

    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Allocation,
    ) -> RendererResult<()> {
        let res = self.delegate.destroy_buffer(device, buffer, memory)?;
        self.buffer_destroyed += 1;
        Ok(res)
    }

    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Allocation)> {
        let res = self.delegate.create_image(device, width, height)?;
        self.image_create += 1;
        Ok(res)
    }

    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Allocation,
    ) -> RendererResult<()> {
        let res = self.delegate.destroy_image(device, image, memory)?;
        self.image_destroyed += 1;
        Ok(res)
    }
}
