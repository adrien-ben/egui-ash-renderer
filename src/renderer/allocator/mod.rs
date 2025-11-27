#[cfg(feature = "simple-allocator")]
mod default;

#[cfg(feature = "gpu-allocator")]
mod gpu;

#[cfg(feature = "vk-mem")]
mod vkmem;

use crate::RendererResult;
use ash::{Device, vk};

pub enum Allocator {
    #[cfg(feature = "simple-allocator")]
    Simple(default::Allocator),
    #[cfg(feature = "gpu-allocator")]
    Gpu(gpu::Allocator),
    #[cfg(feature = "vk-mem")]
    VkMem(vkmem::Allocator),
}

impl Allocator {
    #[cfg(feature = "simple-allocator")]
    pub(super) fn new_simple(memory_properties: ash::vk::PhysicalDeviceMemoryProperties) -> Self {
        Self::Simple(default::Allocator::new(memory_properties))
    }

    #[cfg(feature = "gpu-allocator")]
    pub(super) fn new_gpu(
        allocator: std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>,
    ) -> Self {
        Self::Gpu(gpu::Allocator::new(allocator))
    }

    #[cfg(feature = "vk-mem")]
    pub(super) fn new_vk_mem(allocator: std::sync::Arc<vk_mem::Allocator>) -> Self {
        Self::VkMem(vkmem::Allocator::new(allocator))
    }
}

pub enum Memory {
    #[cfg(feature = "simple-allocator")]
    Simple(default::Memory),
    #[cfg(feature = "gpu-allocator")]
    Gpu(gpu::Memory),
    #[cfg(feature = "vk-mem")]
    VkMem(vkmem::Memory),
}

impl Allocate for Allocator {
    type Memory = Memory;

    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Memory)> {
        match self {
            #[cfg(feature = "simple-allocator")]
            Self::Simple(a) => {
                let (buff, mem) = a.create_buffer(device, size, usage)?;
                Ok((buff, Memory::Simple(mem)))
            }
            #[cfg(feature = "vk-mem")]
            Self::VkMem(a) => {
                let (buff, mem) = a.create_buffer(device, size, usage)?;
                Ok((buff, Memory::VkMem(mem)))
            }
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(a) => {
                let (buff, mem) = a.create_buffer(device, size, usage)?;
                Ok((buff, Memory::Gpu(mem)))
            }
            _ => unreachable!(),
        }
    }

    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &mut Self::Memory,
        data: &[T],
    ) -> RendererResult<()> {
        match (self, memory) {
            #[cfg(feature = "simple-allocator")]
            (Self::Simple(a), Self::Memory::Simple(memory)) => {
                a.update_buffer(device, memory, data)
            }
            #[cfg(feature = "vk-mem")]
            (Self::VkMem(a), Self::Memory::VkMem(memory)) => a.update_buffer(device, memory, data),
            #[cfg(feature = "gpu-allocator")]
            (Self::Gpu(a), Self::Memory::Gpu(memory)) => a.update_buffer(device, memory, data),
            _ => unreachable!(),
        }
    }

    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        match (self, memory) {
            #[cfg(feature = "simple-allocator")]
            (Self::Simple(a), Self::Memory::Simple(memory)) => {
                a.destroy_buffer(device, buffer, memory)
            }
            #[cfg(feature = "vk-mem")]
            (Self::VkMem(a), Self::Memory::VkMem(memory)) => {
                a.destroy_buffer(device, buffer, memory)
            }
            #[cfg(feature = "gpu-allocator")]
            (Self::Gpu(a), Self::Memory::Gpu(memory)) => a.destroy_buffer(device, buffer, memory),
            _ => unreachable!(),
        }
    }

    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Memory)> {
        match self {
            #[cfg(feature = "simple-allocator")]
            Self::Simple(a) => {
                let (img, mem) = a.create_image(device, width, height)?;
                Ok((img, Memory::Simple(mem)))
            }
            #[cfg(feature = "vk-mem")]
            Self::VkMem(a) => {
                let (img, mem) = a.create_image(device, width, height)?;
                Ok((img, Memory::VkMem(mem)))
            }
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(a) => {
                let (img, mem) = a.create_image(device, width, height)?;
                Ok((img, Memory::Gpu(mem)))
            }
            _ => unreachable!(),
        }
    }

    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        match (self, memory) {
            #[cfg(feature = "simple-allocator")]
            (Self::Simple(a), Self::Memory::Simple(memory)) => {
                a.destroy_image(device, image, memory)
            }
            #[cfg(feature = "vk-mem")]
            (Self::VkMem(a), Self::Memory::VkMem(memory)) => a.destroy_image(device, image, memory),
            #[cfg(feature = "gpu-allocator")]
            (Self::Gpu(a), Self::Memory::Gpu(memory)) => a.destroy_image(device, image, memory),
            _ => unreachable!(),
        }
    }
}

/// Base allocator trait for all implementations.
pub trait Allocate {
    type Memory;

    /// Create a Vulkan buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `size` - The size in bytes of the buffer.
    /// * `usage` - The buffer usage flags.
    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Memory)>;

    /// Create a Vulkan image.
    ///
    /// This creates a 2D RGBA8_SRGB image with TRANSFER_DST and SAMPLED flags.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `width` - The width of the image to create.
    /// * `height` - The height of the image to create.
    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Memory)>;

    /// Destroys a buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `buffer` - The buffer to destroy.
    /// * `memory` - The buffer memory to destroy.
    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> RendererResult<()>;

    /// Destroys an image.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `image` - The image to destroy.
    /// * `memory` - The image memory to destroy.
    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> RendererResult<()>;

    /// Update buffer data
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `memory` - The memory of the buffer to update.
    /// * `data` - The data to update the buffer with.
    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &mut Self::Memory,
        data: &[T],
    ) -> RendererResult<()>;
}
