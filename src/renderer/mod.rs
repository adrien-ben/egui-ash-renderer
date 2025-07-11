mod allocator;
pub mod vulkan;

use std::collections::HashMap;

use crate::RendererError;
use ash::{Device, vk};
use egui::{
    ClippedPrimitive, ImageData, TextureId,
    epaint::{ImageDelta, Primitive},
};
use mesh::*;
use vulkan::*;

use self::allocator::Allocator;

#[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
use ash::Instance;

#[cfg(feature = "gpu-allocator")]
use {
    gpu_allocator::vulkan::Allocator as GpuAllocator,
    std::sync::{Arc, Mutex},
};

#[cfg(feature = "vk-mem")]
use {
    std::sync::{Arc, Mutex},
    vk_mem::Allocator as VkMemAllocator,
};

/// Convenient return type for function that can return a [`RendererError`].
///
/// [`RendererError`]: enum.RendererError.html
pub type RendererResult<T> = Result<T, RendererError>;

const MAX_TEXTURE_COUNT: u32 = 1024; // TODO: constant max size or user defined ?

/// Optional parameters of the renderer.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// The number of in flight frames of the application.
    pub in_flight_frames: usize,
    /// If true enables depth test when rendering.
    pub enable_depth_test: bool,
    /// If true enables depth writes when rendering.
    ///
    /// Note that depth writes are always disabled when enable_depth_test is false.
    /// See <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPipelineDepthStencilStateCreateInfo.html>
    pub enable_depth_write: bool,
    /// Is the target framebuffer sRGB.
    ///
    /// If not, the fragment shader converts colors to sRGB, otherwise it outputs color in linear space.
    pub srgb_framebuffer: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            in_flight_frames: 1,
            enable_depth_test: false,
            enable_depth_write: false,
            srgb_framebuffer: false,
        }
    }
}

/// `dynamic-rendering` feature related params
#[cfg(feature = "dynamic-rendering")]
#[derive(Debug, Clone, Copy)]
pub struct DynamicRendering {
    pub color_attachment_format: vk::Format,
    pub depth_attachment_format: Option<vk::Format>,
}

/// Vulkan renderer for egui.
///
/// It records rendering command to the provided command buffer at each call to [`cmd_draw`].
///
/// The renderer holds a set of vertex/index buffers per in flight frames. Vertex and index buffers
/// are resized at each call to [`cmd_draw`] if draw data does not fit.
///
/// [`cmd_draw`]: #method.cmd_draw
pub struct Renderer {
    device: Device,
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    managed_textures: HashMap<TextureId, Texture>,
    textures: HashMap<TextureId, vk::DescriptorSet>,
    next_user_texture_id: u64,
    options: Options,
    frames: Option<Frames>,
}

impl Renderer {
    /// Create a renderer using the default allocator.
    ///
    /// At initialization all Vulkan resources are initialized. Vertex and index buffers are not created yet.
    ///
    /// # Arguments
    ///
    /// * `instance` - A reference to a Vulkan instance.
    /// * `physical_device` - A Vulkan physical device.
    /// * `device` - A Vulkan device.
    /// * `render_pass` - *without dynamic-rendering feature* - The render pass used to render the gui.
    /// * `dynamic_rendering` - *with dynamic-rendering feature* - Dynamic rendeing parameters
    /// * `options` - Rendering options.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If the number of in flight frame in incorrect.
    /// * [`RendererError`] - If any Vulkan or io error is encountered during initialization.
    #[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
    pub fn with_default_allocator(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        #[cfg(not(feature = "dynamic-rendering"))] render_pass: vk::RenderPass,
        #[cfg(feature = "dynamic-rendering")] dynamic_rendering: DynamicRendering,
        options: Options,
    ) -> RendererResult<Self> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self::from_allocator(
            device,
            Allocator::new(memory_properties),
            #[cfg(not(feature = "dynamic-rendering"))]
            render_pass,
            #[cfg(feature = "dynamic-rendering")]
            dynamic_rendering,
            options,
        )
    }

    /// Create a renderer using gpu-allocator.
    ///
    /// At initialization all Vulkan resources are initialized. Vertex and index buffers are not created yet.
    ///
    /// # Arguments
    ///
    /// * `gpu_allocator` - The allocator that will be used to allocator buffer and image memory.
    /// * `device` - A Vulkan device.
    /// * `render_pass` - *without dynamic-rendering feature* - The render pass used to render the gui.
    /// * `dynamic_rendering` - *with dynamic-rendering feature* - Dynamic rendeing parameters
    /// * `options` - Rendering options.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If the number of in flight frame in incorrect.
    /// * [`RendererError`] - If any Vulkan or io error is encountered during initialization.
    #[cfg(feature = "gpu-allocator")]
    pub fn with_gpu_allocator(
        gpu_allocator: Arc<Mutex<GpuAllocator>>,
        device: Device,
        #[cfg(not(feature = "dynamic-rendering"))] render_pass: vk::RenderPass,
        #[cfg(feature = "dynamic-rendering")] dynamic_rendering: DynamicRendering,
        options: Options,
    ) -> RendererResult<Self> {
        Self::from_allocator(
            device,
            Allocator::new(gpu_allocator),
            #[cfg(not(feature = "dynamic-rendering"))]
            render_pass,
            #[cfg(feature = "dynamic-rendering")]
            dynamic_rendering,
            options,
        )
    }

    /// Create a renderer using vk-mem.
    ///
    /// At initialization all Vulkan resources are initialized. Vertex and index buffers are not created yet.
    ///
    /// # Arguments
    ///
    /// * `vk_mem_allocator` - The allocator that will be used to allocator buffer and image memory.
    /// * `device` - A Vulkan device.
    /// * `render_pass` - *without dynamic-rendering feature* - The render pass used to render the gui.
    /// * `dynamic_rendering` - *with dynamic-rendering feature* - Dynamic rendeing parameters
    /// * `options` - Rendering options.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If the number of in flight frame in incorrect.
    /// * [`RendererError`] - If any Vulkan or io error is encountered during initialization.
    #[cfg(feature = "vk-mem")]
    pub fn with_vk_mem_allocator(
        vk_mem_allocator: Arc<Mutex<VkMemAllocator>>,
        device: Device,
        #[cfg(not(feature = "dynamic-rendering"))] render_pass: vk::RenderPass,
        #[cfg(feature = "dynamic-rendering")] dynamic_rendering: DynamicRendering,
        options: Options,
    ) -> RendererResult<Self> {
        Self::from_allocator(
            device,
            Allocator::new(vk_mem_allocator),
            #[cfg(not(feature = "dynamic-rendering"))]
            render_pass,
            #[cfg(feature = "dynamic-rendering")]
            dynamic_rendering,
            options,
        )
    }

    fn from_allocator(
        device: Device,
        allocator: Allocator,
        #[cfg(not(feature = "dynamic-rendering"))] render_pass: vk::RenderPass,
        #[cfg(feature = "dynamic-rendering")] dynamic_rendering: DynamicRendering,
        options: Options,
    ) -> RendererResult<Self> {
        log::debug!("Creating egui renderer with options {options:?}");

        if options.in_flight_frames == 0 {
            return Err(RendererError::Init(String::from(
                "'in_flight_frames' parameter should be at least one",
            )));
        }

        // Descriptor set layout
        let descriptor_set_layout = create_vulkan_descriptor_set_layout(&device)?;

        // Pipeline and layout
        let pipeline_layout = create_vulkan_pipeline_layout(&device, descriptor_set_layout)?;
        let pipeline = create_vulkan_pipeline(
            &device,
            pipeline_layout,
            #[cfg(not(feature = "dynamic-rendering"))]
            render_pass,
            #[cfg(feature = "dynamic-rendering")]
            dynamic_rendering,
            options,
        )?;

        // Descriptor pool
        let descriptor_pool = create_vulkan_descriptor_pool(&device, MAX_TEXTURE_COUNT)?;

        // Textures
        let managed_textures = HashMap::new();
        let textures = HashMap::new();

        Ok(Self {
            device,
            allocator,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            managed_textures,
            next_user_texture_id: 0,
            textures,
            options,
            frames: None,
        })
    }

    /// Change the render pass to render to.
    ///
    /// Useful if you need to render to a new render pass.
    /// It will rebuild the graphics pipeline from scratch so it is an expensive operation.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass used to render the gui.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered during pipeline creation.
    #[cfg(not(feature = "dynamic-rendering"))]
    pub fn set_render_pass(&mut self, render_pass: vk::RenderPass) -> RendererResult<()> {
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = create_vulkan_pipeline(
            &self.device,
            self.pipeline_layout,
            render_pass,
            self.options,
        )?;
        Ok(())
    }

    /// Change the dynamic rendering parameters.
    ///
    /// Useful if you need to render to a target of with another color/depth format.
    /// It will rebuild the graphics pipeline from scratch so it is an expensive operation.
    ///
    /// # Arguments
    ///
    /// * `dynamic_rendering` - The new dynamic rendering parameters.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered during pipeline creation.
    #[cfg(feature = "dynamic-rendering")]
    pub fn set_dynamic_rendering(
        &mut self,
        dynamic_rendering: DynamicRendering,
    ) -> RendererResult<()> {
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = create_vulkan_pipeline(
            &self.device,
            self.pipeline_layout,
            dynamic_rendering,
            self.options,
        )?;
        Ok(())
    }

    /// Free egui managed textures.
    ///
    /// You should pass the list of textures detla contained in the [`egui::TexturesDelta::set`].
    /// This method should be called _before_ the frame starts rendering.
    ///
    /// # Arguments
    ///
    /// * `ids` - The list of ids of textures to free.
    /// * `queue` - The queue used to copy image data on the GPU.
    /// * `command_pool` - A Vulkan command pool used to allocate command buffers to upload textures to the gpu.
    /// * `textures_delta` - The modifications to apply to the textures.
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered during pipeline creation.
    pub fn set_textures(
        &mut self,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        textures_delta: &[(TextureId, ImageDelta)],
    ) -> RendererResult<()> {
        log::trace!("Setting {} textures", textures_delta.len());
        for (id, delta) in textures_delta {
            let (width, height, data) = match &delta.image {
                ImageData::Color(image) => {
                    let w = image.width() as u32;
                    let h = image.height() as u32;
                    let data = image
                        .pixels
                        .iter()
                        .flat_map(|c| c.to_array())
                        .collect::<Vec<_>>();

                    (w, h, data)
                }
            };

            if let Some([offset_x, offset_y]) = delta.pos {
                log::trace!("Updating texture {id:?}");

                let texture = self
                    .managed_textures
                    .get_mut(id)
                    .ok_or(RendererError::BadTexture(*id))?;

                texture.update(
                    &self.device,
                    queue,
                    command_pool,
                    &mut self.allocator,
                    vk::Rect2D {
                        offset: vk::Offset2D {
                            x: offset_x as _,
                            y: offset_y as _,
                        },
                        extent: vk::Extent2D { width, height },
                    },
                    data.as_slice(),
                )?;
            } else {
                log::trace!("Adding texture {id:?}");

                let texture = Texture::from_rgba8(
                    &self.device,
                    queue,
                    command_pool,
                    &mut self.allocator,
                    width,
                    height,
                    data.as_slice(),
                )?;

                let set = create_vulkan_descriptor_set(
                    &self.device,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    texture.image_view,
                    texture.sampler,
                )?;

                if let Some(previous) = self.managed_textures.insert(*id, texture) {
                    previous.destroy(&self.device, &mut self.allocator)?;
                }
                if let Some(previous) = self.textures.insert(*id, set) {
                    unsafe {
                        self.device
                            .free_descriptor_sets(self.descriptor_pool, &[previous])?
                    };
                }
            }
        }

        Ok(())
    }

    /// Free egui managed textures.
    ///
    /// You should pass the list of ids contained in the [`egui::TexturesDelta::free`].
    /// This method should be called _after_ the frame is done rendering.
    ///
    /// # Arguments
    ///
    /// * `ids` - The list of ids of textures to free.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered when free the texture.
    pub fn free_textures(&mut self, ids: &[TextureId]) -> RendererResult<()> {
        log::trace!("Freeing {} textures", ids.len());
        for id in ids {
            if let Some(texture) = self.managed_textures.remove(id) {
                texture.destroy(&self.device, &mut self.allocator)?;
            }
            if let Some(set) = self.textures.remove(id) {
                unsafe {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[set])?
                };
            }
        }

        Ok(())
    }

    /// Add a user managed texture used by egui.
    ///
    /// The descriptors set passed in this method are managed by the used and *will not* be freed by the renderer.
    /// This method will return a [`egui::TextureId`] which can then be used in a [`egui::Image`].
    ///
    /// # Arguments
    ///
    /// * `set` - The descpritor set referencing the texture to display.
    ///
    /// # Caveat
    ///
    /// Provided `vk::DescriptorSet`s must be created with a descriptor set layout that is compatible with the one used by the renderer.
    /// See [Pipeline Layout Compatibility](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#descriptorsets-compatibility).
    pub fn add_user_texture(&mut self, set: vk::DescriptorSet) -> TextureId {
        let id = TextureId::User(self.next_user_texture_id);
        self.next_user_texture_id += 1;
        self.textures.insert(id, set);

        id
    }

    /// Remove a user managed texture.
    ///
    /// This *does not* free the resources, it just _forgets_ about the texture.
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the texture to remove.
    pub fn remove_user_texture(&mut self, id: TextureId) {
        self.textures.remove(&id);
    }

    /// Record commands to render the [`egui::Ui`].
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - The Vulkan command buffer that command will be recorded to.
    /// * `extent` - The extent of the surface to render to.
    /// * `pixel_per_point` - The number of physical pixels per point. See [`egui::FullOutput::pixels_per_point`].
    /// * `primitives` - The primitives to render. See [`egui::Context::tessellate`].
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered when recording.
    pub fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        pixels_per_point: f32,
        primitives: &[ClippedPrimitive],
    ) -> RendererResult<()> {
        if primitives.is_empty() {
            return Ok(());
        }

        if self.frames.is_none() {
            self.frames.replace(Frames::new(
                &self.device,
                &mut self.allocator,
                primitives,
                self.options.in_flight_frames,
            )?);
        }

        let mesh = self.frames.as_mut().unwrap().next();
        mesh.update(&self.device, &mut self.allocator, primitives)?;

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            )
        };

        let screen_width = extent.width as f32;
        let screen_height = extent.height as f32;

        unsafe {
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    width: screen_width,
                    height: screen_height,
                    max_depth: 1.0,
                    ..Default::default()
                }],
            )
        };

        // Ortho projection
        let projection = orthographic_vk(
            0.0,
            screen_width / pixels_per_point,
            0.0,
            -(screen_height / pixels_per_point),
            -1.0,
            1.0,
        );
        unsafe {
            let push = any_as_u8_slice(&projection);
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push,
            )
        };

        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                mesh.indices,
                0,
                vk::IndexType::UINT32,
            )
        };

        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertices], &[0])
        };

        let mut index_offset = 0u32;
        let mut vertex_offset = 0i32;
        let mut current_texture_id: Option<TextureId> = None;

        for p in primitives {
            let clip_rect = p.clip_rect;
            match &p.primitive {
                Primitive::Mesh(m) => {
                    let clip_x = clip_rect.min.x * pixels_per_point;
                    let clip_y = clip_rect.min.y * pixels_per_point;
                    let clip_w = clip_rect.max.x * pixels_per_point - clip_x;
                    let clip_h = clip_rect.max.y * pixels_per_point - clip_y;

                    let scissors = [vk::Rect2D {
                        offset: vk::Offset2D {
                            x: (clip_x as i32).max(0),
                            y: (clip_y as i32).max(0),
                        },
                        extent: vk::Extent2D {
                            width: clip_w.min(screen_width) as _,
                            height: clip_h.min(screen_height) as _,
                        },
                    }];

                    unsafe {
                        self.device.cmd_set_scissor(command_buffer, 0, &scissors);
                    }

                    if Some(m.texture_id) != current_texture_id {
                        let descriptor_set = *self
                            .textures
                            .get(&m.texture_id)
                            .ok_or(RendererError::BadTexture(m.texture_id))?;

                        unsafe {
                            self.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[descriptor_set],
                                &[],
                            )
                        };
                        current_texture_id = Some(m.texture_id);
                    }

                    let index_count = m.indices.len() as u32;
                    unsafe {
                        self.device.cmd_draw_indexed(
                            command_buffer,
                            index_count,
                            1,
                            index_offset,
                            vertex_offset,
                            0,
                        )
                    };

                    index_offset += index_count;
                    vertex_offset += m.vertices.len() as i32;
                }
                Primitive::Callback(_) => {
                    log::warn!("Callback primitives not yet supported")
                }
            }
        }

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        log::debug!("Destroying egui renderer");
        let device = &self.device;

        unsafe {
            if let Some(frames) = self.frames.take() {
                frames
                    .destroy(device, &mut self.allocator)
                    .expect("Failed to destroy frame data");
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            for (_, t) in self.managed_textures.drain() {
                t.destroy(device, &mut self.allocator)
                    .expect("Failed to destroy texture");
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// Structure holding data for all frames in flight.
struct Frames {
    index: usize,
    count: usize,
    meshes: Vec<Mesh>,
}

impl Frames {
    fn new(
        device: &Device,
        allocator: &mut Allocator,
        primitives: &[ClippedPrimitive],
        count: usize,
    ) -> RendererResult<Self> {
        let meshes = (0..count)
            .map(|_| Mesh::new(device, allocator, primitives))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            index: 0,
            count,
            meshes,
        })
    }

    fn next(&mut self) -> &mut Mesh {
        let result = &mut self.meshes[self.index];
        self.index = (self.index + 1) % self.count;
        result
    }

    fn destroy(self, device: &Device, allocator: &mut Allocator) -> RendererResult<()> {
        for mesh in self.meshes.into_iter() {
            mesh.destroy(device, allocator)?;
        }
        Ok(())
    }
}

mod mesh {

    use super::allocator::{Allocate, Allocator, Memory};
    use super::vulkan::*;
    use crate::RendererResult;
    use ash::{Device, vk};
    use egui::ClippedPrimitive;
    use egui::epaint::{Primitive, Vertex};
    use std::mem::size_of;

    /// Vertex and index buffer resources for one frame in flight.
    pub struct Mesh {
        pub vertices: vk::Buffer,
        vertices_mem: Memory,
        vertex_count: usize,
        pub indices: vk::Buffer,
        indices_mem: Memory,
        index_count: usize,
    }

    impl Mesh {
        pub fn new(
            device: &Device,
            allocator: &mut Allocator,
            primitives: &[ClippedPrimitive],
        ) -> RendererResult<Self> {
            let vertices = create_vertices(primitives);
            let vertex_count = vertices.len();
            let indices = create_indices(primitives);
            let index_count = indices.len();

            // Create a vertex buffer
            let (vertices, vertices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &vertices,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            // Create an index buffer
            let (indices, indices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            Ok(Mesh {
                vertices,
                vertices_mem,
                vertex_count,
                indices,
                indices_mem,
                index_count,
            })
        }

        pub fn update(
            &mut self,
            device: &Device,
            allocator: &mut Allocator,
            primitives: &[ClippedPrimitive],
        ) -> RendererResult<()> {
            let vertices = create_vertices(primitives);
            if vertices.len() > self.vertex_count {
                log::trace!("Resizing vertex buffers");

                let vertex_count = vertices.len();
                let size = vertex_count * size_of::<Vertex>();
                let (vertices, vertices_mem) =
                    allocator.create_buffer(device, size, vk::BufferUsageFlags::VERTEX_BUFFER)?;

                self.vertex_count = vertex_count;

                let old_vertices = self.vertices;
                self.vertices = vertices;

                let old_vertices_mem = std::mem::replace(&mut self.vertices_mem, vertices_mem);

                allocator.destroy_buffer(device, old_vertices, old_vertices_mem)?;
            }
            allocator.update_buffer(device, &mut self.vertices_mem, &vertices)?;

            let indices = create_indices(primitives);
            if indices.len() > self.index_count {
                log::trace!("Resizing index buffers");

                let index_count = indices.len();
                let size = index_count * size_of::<u32>();
                let (indices, indices_mem) =
                    allocator.create_buffer(device, size, vk::BufferUsageFlags::INDEX_BUFFER)?;

                self.index_count = index_count;

                let old_indices = self.indices;
                self.indices = indices;

                let old_indices_mem = std::mem::replace(&mut self.indices_mem, indices_mem);

                allocator.destroy_buffer(device, old_indices, old_indices_mem)?;
            }
            allocator.update_buffer(device, &mut self.indices_mem, &indices)?;

            Ok(())
        }

        pub fn destroy(self, device: &Device, allocator: &mut Allocator) -> RendererResult<()> {
            allocator.destroy_buffer(device, self.vertices, self.vertices_mem)?;
            allocator.destroy_buffer(device, self.indices, self.indices_mem)?;
            Ok(())
        }
    }

    fn create_vertices(primitives: &[ClippedPrimitive]) -> Vec<Vertex> {
        let vertex_count = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(m) => m.vertices.len(),
                _ => 0,
            })
            .sum();

        let mut vertices = Vec::with_capacity(vertex_count);
        for p in primitives {
            if let Primitive::Mesh(m) = &p.primitive {
                vertices.extend_from_slice(&m.vertices);
            }
        }
        vertices
    }

    fn create_indices(primitives: &[ClippedPrimitive]) -> Vec<u32> {
        let index_count = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(m) => m.indices.len(),
                _ => 0,
            })
            .sum();

        let mut indices = Vec::with_capacity(index_count);
        for p in primitives {
            if let Primitive::Mesh(m) = &p.primitive {
                indices.extend_from_slice(&m.indices);
            }
        }

        indices
    }
}

/// Orthographic projection matrix for use with Vulkan.
///
/// This matrix is meant to be used when the source coordinate space is right-handed and y-up
/// (the standard computer graphics coordinate space)and the destination space is right-handed
/// and y-down, with Z (depth) clip extending from 0.0 (close) to 1.0 (far).
///
/// from: https://github.com/fu5ha/ultraviolet (to limit dependencies)
#[inline]
pub fn orthographic_vk(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [f32; 16] {
    let rml = right - left;
    let rpl = right + left;
    let tmb = top - bottom;
    let tpb = top + bottom;
    let fmn = far - near;

    #[rustfmt::skip]
    let res = [
        2.0 / rml, 0.0, 0.0, 0.0,
        0.0, -2.0 / tmb, 0.0, 0.0,
        0.0, 0.0, -1.0 / fmn, 0.0,
        -(rpl / rml), -(tpb / tmb), -(near / fmn), 1.0
    ];

    res
}
