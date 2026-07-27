mod default;

#[cfg(feature = "gpu-allocator")]
mod gpu;

#[cfg(feature = "vk-mem")]
mod vkmem;

pub use default::Allocator as DefaultAllocator;

#[cfg(feature = "gpu-allocator")]
pub use gpu::Allocator as GpuAllocator;

#[cfg(feature = "vk-mem")]
pub use vkmem::Allocator as VkMemAllocator;

use crate::RendererResult;
use ash::{Device, vk};

/// Base allocator trait for all implementations.
pub trait Allocator {
    type Allocation;

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
    ) -> RendererResult<(vk::Buffer, Self::Allocation)>;

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
    ) -> RendererResult<(vk::Image, Self::Allocation)>;

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
        memory: Self::Allocation,
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
        memory: Self::Allocation,
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
        memory: &mut Self::Allocation,
        data: &[T],
    ) -> RendererResult<()>;
}
