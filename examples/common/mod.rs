pub mod vulkan;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn};
use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use egui::{ClippedPrimitive, Context, TextureId, ViewportId};
use egui_ash_renderer::{Options, Renderer};
use egui_winit::State;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    error::Error,
    ffi::{CStr, CString},
    os::raw::c_void,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
#[cfg(feature = "gpu-allocator")]
use {
    gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc},
    std::sync::{Arc, Mutex},
};
#[cfg(feature = "vk-mem")]
use {
    std::sync::{Arc, Mutex},
    vk_mem::{Allocator, AllocatorCreateInfo},
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 1024;

pub trait App {
    fn title() -> &'static str;
    fn new(app: &mut System) -> Self;
    fn build_ui(&mut self, ctx: &Context);
    fn clean(&mut self, vulkan_ctx: &VulkanContext) {
        let _ = vulkan_ctx;
    }
}

pub fn run_app<A: App>() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app: AppWrapper<A> = AppWrapper {
        title: A::title(),
        window: None,
        app: None,
        egui_app: None,
    };

    event_loop.run_app(&mut app)?;

    Ok(())
}

struct AppWrapper<A: App> {
    title: &'static str,
    window: Option<Window>,
    app: Option<System>,
    egui_app: Option<A>,
}

impl<A: App> ApplicationHandler for AppWrapper<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(self.title, event_loop).unwrap();
        let mut app = System::new(self.title, &window).unwrap();
        let egui_app = A::new(&mut app);

        self.window = Some(window);
        self.app = Some(app);
        self.egui_app = Some(egui_app)
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        self.app.as_mut().unwrap().draw(
            self.egui_app.as_mut().unwrap(),
            self.window.as_ref().unwrap(),
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        self.app.as_mut().unwrap().on_window_event(
            &event,
            self.window.as_ref().unwrap(),
            event_loop,
        );
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.app
            .as_mut()
            .unwrap()
            .on_exit(self.egui_app.as_mut().unwrap());
    }
}

pub struct System {
    command_buffer: vk::CommandBuffer,
    swapchain: Swapchain,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,

    run: bool,
    dirty_swapchain: bool,

    pub egui_ctx: Context,
    pub egui_winit: State,
    pub renderer: Renderer,

    textures_to_free: Option<Vec<TextureId>>,

    pub vulkan_context: VulkanContext,
}

impl System {
    pub fn new(title: &str, window: &Window) -> Result<Self, Box<dyn Error>> {
        log::info!("Create application");

        let vulkan_context = VulkanContext::new(&window, title)?;

        let command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(vulkan_context.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            unsafe {
                vulkan_context
                    .device
                    .allocate_command_buffers(&allocate_info)?[0]
            }
        };

        let swapchain = Swapchain::new(&vulkan_context)?;

        // Semaphore use for presentation
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe {
                vulkan_context
                    .device
                    .create_semaphore(&semaphore_info, None)?
            }
        };
        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            unsafe {
                vulkan_context
                    .device
                    .create_semaphore(&semaphore_info, None)?
            }
        };
        let fence = {
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            unsafe { vulkan_context.device.create_fence(&fence_info, None)? }
        };

        let egui_ctx = Context::default();
        egui_extras::install_image_loaders(&egui_ctx);
        let egui_winit = State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );

        #[cfg(feature = "gpu-allocator")]
        let renderer = {
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: vulkan_context.instance.clone(),
                device: vulkan_context.device.clone(),
                physical_device: vulkan_context.physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })?;

            Renderer::with_gpu_allocator(
                Arc::new(Mutex::new(allocator)),
                vulkan_context.device.clone(),
                swapchain.render_pass,
                Options {
                    srgb_framebuffer: true,
                    ..Default::default()
                },
            )?
        };

        #[cfg(feature = "vk-mem")]
        let renderer = {
            let allocator = {
                let allocator_create_info = AllocatorCreateInfo::new(
                    &vulkan_context.instance,
                    &vulkan_context.device,
                    vulkan_context.physical_device,
                );

                unsafe { Allocator::new(allocator_create_info)? }
            };

            Renderer::with_vk_mem_allocator(
                Arc::new(Mutex::new(allocator)),
                vulkan_context.device.clone(),
                swapchain.render_pass,
                Options {
                    srgb_framebuffer: true,
                    ..Default::default()
                },
            )?
        };

        #[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
        let renderer = Renderer::with_default_allocator(
            &vulkan_context.instance,
            vulkan_context.physical_device,
            vulkan_context.device.clone(),
            swapchain.render_pass,
            Options {
                srgb_framebuffer: true,
                ..Default::default()
            },
        )?;

        Ok(Self {
            vulkan_context,
            command_buffer,
            swapchain,
            image_available_semaphore,
            render_finished_semaphore,
            fence,

            run: true,
            dirty_swapchain: false,

            egui_ctx,
            egui_winit,
            renderer,

            textures_to_free: None,
        })
    }

    fn draw<A: App>(&mut self, egui_app: &mut A, window: &Window) {
        // If swapchain must be recreated wait for windows to not be minimized anymore
        if self.dirty_swapchain {
            let PhysicalSize { width, height } = window.inner_size();
            if width > 0 && height > 0 {
                self.swapchain
                    .recreate(&self.vulkan_context)
                    .expect("Failed to recreate swapchain");
                self.renderer
                    .set_render_pass(self.swapchain.render_pass)
                    .expect("Failed to rebuild renderer pipeline");
                self.dirty_swapchain = false;
            } else {
                return;
            }
        }

        if !self.run {
            return;
        }

        unsafe {
            self.vulkan_context
                .device
                .wait_for_fences(&[self.fence], true, std::u64::MAX)
                .expect("Failed to wait ")
        };

        // Free last frames textures after the previous frame is done rendering
        if let Some(textures) = self.textures_to_free.take() {
            self.renderer
                .free_textures(&textures)
                .expect("Failed to free textures");
        }

        // Generate UI
        let raw_input = self.egui_winit.take_egui_input(&window);

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.egui_ctx.run(raw_input, |ctx| {
            egui_app.build_ui(ctx);
        });

        self.egui_winit
            .handle_platform_output(&window, platform_output);

        if !textures_delta.free.is_empty() {
            self.textures_to_free = Some(textures_delta.free.clone());
        }

        if !textures_delta.set.is_empty() {
            self.renderer
                .set_textures(
                    self.vulkan_context.graphics_queue,
                    self.vulkan_context.command_pool,
                    textures_delta.set.as_slice(),
                )
                .expect("Failed to update texture");
        }

        let clipped_primitives = self.egui_ctx.tessellate(shapes, pixels_per_point);

        // Drawing the frame
        let next_image_result = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.khr,
                std::u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match next_image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_swapchain = true;
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe {
            self.vulkan_context
                .device
                .reset_fences(&[self.fence])
                .expect("Failed to reset fences")
        };

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.image_available_semaphore];
        let signal_semaphores = [self.render_finished_semaphore];

        // Re-record commands to draw geometry
        record_command_buffers(
            &self.vulkan_context.device,
            self.vulkan_context.command_pool,
            self.command_buffer,
            self.swapchain.framebuffers[image_index as usize],
            self.swapchain.render_pass,
            self.swapchain.extent,
            pixels_per_point,
            &mut self.renderer,
            &clipped_primitives,
        )
        .expect("Failed to record command buffer");

        let command_buffers = [self.command_buffer];
        let submit_info = [vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)];
        unsafe {
            self.vulkan_context
                .device
                .queue_submit(self.vulkan_context.graphics_queue, &submit_info, self.fence)
                .expect("Failed to submit work to gpu.")
        };

        let swapchains = [self.swapchain.khr];
        let images_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);

        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.vulkan_context.present_queue, &present_info)
        };
        match present_result {
            Ok(is_suboptimal) if is_suboptimal => {
                self.dirty_swapchain = true;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_swapchain = true;
            }
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => {}
        }
    }

    fn on_window_event(
        &mut self,
        event: &WindowEvent,
        window: &Window,
        event_loop: &ActiveEventLoop,
    ) {
        let _ = self.egui_winit.on_window_event(&window, &event);

        match event {
            // Resizing
            WindowEvent::Resized(new_size) => {
                log::debug!("Window was resized. New size is {new_size:?}");
                self.dirty_swapchain = true;
            }
            // Exit
            WindowEvent::CloseRequested => {
                self.run = false;
                event_loop.exit();
            }
            _ => (),
        }
    }

    fn on_exit<A: App>(&mut self, egui_app: &mut A) {
        log::info!("Stopping application");

        unsafe {
            self.vulkan_context
                .device
                .device_wait_idle()
                .expect("Failed to wait for graphics device to idle.");

            egui_app.clean(&self.vulkan_context);

            self.vulkan_context.device.destroy_fence(self.fence, None);
            self.vulkan_context
                .device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.vulkan_context
                .device
                .destroy_semaphore(self.render_finished_semaphore, None);

            self.swapchain.destroy(&self.vulkan_context);

            self.vulkan_context
                .device
                .free_command_buffers(self.vulkan_context.command_pool, &[self.command_buffer]);
        }
    }
}

pub struct VulkanContext {
    _entry: Entry,
    pub instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
}

impl VulkanContext {
    pub fn new(window: &Window, name: &str) -> Result<Self, Box<dyn Error>> {
        // Vulkan instance
        let entry = Entry::linked();
        let (instance, debug_utils, debug_utils_messenger) =
            create_vulkan_instance(&entry, window, name)?;

        // Vulkan surface
        let surface = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };

        // Vulkan physical device and queue families indices (graphics and present)
        let (physical_device, graphics_q_index, present_q_index) =
            create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
                &instance,
                &surface,
                surface_khr,
            )?;

        // Vulkan logical device and queues
        let (device, graphics_queue, present_queue) =
            create_vulkan_device_and_graphics_and_present_qs(
                &instance,
                physical_device,
                graphics_q_index,
                present_q_index,
            )?;

        // Command pool & buffer
        let command_pool = {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_q_index)
                .flags(vk::CommandPoolCreateFlags::empty());
            unsafe { device.create_command_pool(&command_pool_info, None)? }
        };

        Ok(Self {
            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            surface,
            surface_khr,
            physical_device,
            graphics_q_index,
            present_q_index,
            device,
            graphics_queue,
            present_queue,
            command_pool,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        log::debug!("Destroying Vulkan Context");
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Swapchain {
    loader: swapchain::Device,
    extent: vk::Extent2D,
    khr: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Swapchain {
    fn new(vulkan_context: &VulkanContext) -> Result<Self, Box<dyn Error>> {
        // Swapchain
        let (loader, khr, extent, format, images, image_views) =
            create_vulkan_swapchain(&vulkan_context)?;

        // Renderpass
        let render_pass = create_vulkan_render_pass(&vulkan_context.device, format)?;

        // Framebuffers
        let framebuffers =
            create_vulkan_framebuffers(&vulkan_context.device, render_pass, extent, &image_views)?;

        Ok(Self {
            loader,
            extent,
            khr,
            images,
            image_views,
            render_pass,
            framebuffers,
        })
    }

    fn recreate(&mut self, vulkan_context: &VulkanContext) -> Result<(), Box<dyn Error>> {
        log::debug!("Recreating the swapchain");

        unsafe { vulkan_context.device.device_wait_idle()? };

        self.destroy(vulkan_context);

        // Swapchain
        let (loader, khr, extent, format, images, image_views) =
            create_vulkan_swapchain(vulkan_context)?;

        // Renderpass
        let render_pass = create_vulkan_render_pass(&vulkan_context.device, format)?;

        // Framebuffers
        let framebuffers =
            create_vulkan_framebuffers(&vulkan_context.device, render_pass, extent, &image_views)?;

        self.loader = loader;
        self.extent = extent;
        self.khr = khr;
        self.images = images;
        self.image_views = image_views;
        self.render_pass = render_pass;
        self.framebuffers = framebuffers;

        Ok(())
    }

    fn destroy(&mut self, vulkan_context: &VulkanContext) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|fb| vulkan_context.device.destroy_framebuffer(*fb, None));
            self.framebuffers.clear();
            vulkan_context
                .device
                .destroy_render_pass(self.render_pass, None);
            self.image_views
                .iter()
                .for_each(|v| vulkan_context.device.destroy_image_view(*v, None));
            self.image_views.clear();
            self.loader.destroy_swapchain(self.khr, None);
        }
    }
}

fn create_window(title: &str, event_loop: &ActiveEventLoop) -> Result<Window, Box<dyn Error>> {
    log::debug!("Creating window");

    let window = event_loop.create_window(
        Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
    )?;

    Ok(window)
}

fn create_vulkan_instance(
    entry: &Entry,
    window: &Window,
    title: &str,
) -> Result<(Instance, debug_utils::Instance, vk::DebugUtilsMessengerEXT), Box<dyn Error>> {
    log::debug!("Creating vulkan instance");
    // Vulkan instance
    let app_name = CString::new(title)?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name.as_c_str())
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let mut extension_names =
        ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
    extension_names.push(debug_utils::NAME.as_ptr());

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(create_flags)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

    // Vulkan debug report
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = debug_utils::Instance::new(entry, &instance);
    let debug_utils_messenger =
        unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };

    Ok((instance, debug_utils, debug_utils_messenger))
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

        let message = CStr::from_ptr((*p_callback_data).p_message);
        match flag {
            Flag::VERBOSE => log::debug!("{typ:?} - {message:?}"),
            Flag::INFO => log::info!("{typ:?} - {message:?}"),
            Flag::WARNING => log::warn!("{typ:?} - {message:?}"),
            _ => log::error!("{typ:?} - {message:?}"),
        }
        vk::FALSE
    }
}

fn create_vulkan_physical_device_and_get_graphics_and_present_qs_indices(
    instance: &Instance,
    surface: &surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), Box<dyn Error>> {
    log::debug!("Creating vulkan physical device");
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut graphics = None;
    let mut present = None;
    let device = devices
        .into_iter()
        .find(|device| {
            let device = *device;

            // Does device supports graphics and present queues
            let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
            for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
                let index = index as u32;
                graphics = None;
                present = None;

                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && graphics.is_none()
                {
                    graphics = Some(index);
                }

                let present_support = unsafe {
                    surface
                        .get_physical_device_surface_support(device, index, surface_khr)
                        .expect("Failed to get surface support")
                };
                if present_support && present.is_none() {
                    present = Some(index);
                }

                if graphics.is_some() && present.is_some() {
                    break;
                }
            }

            // Does device support desired extensions
            let extension_props = unsafe {
                instance
                    .enumerate_device_extension_properties(device)
                    .expect("Failed to get device ext properties")
            };
            let extention_support = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                swapchain::NAME == name
            });

            // Does the device have available formats for the given surface
            let formats = unsafe {
                surface
                    .get_physical_device_surface_formats(device, surface_khr)
                    .expect("Failed to get physical device surface formats")
            };

            // Does the device have available present modes for the given surface
            let present_modes = unsafe {
                surface
                    .get_physical_device_surface_present_modes(device, surface_khr)
                    .expect("Failed to get physical device surface present modes")
            };

            graphics.is_some()
                && present.is_some()
                && extention_support
                && !formats.is_empty()
                && !present_modes.is_empty()
        })
        .expect("Could not find a suitable device");

    unsafe {
        let props = instance.get_physical_device_properties(device);
        let device_name = CStr::from_ptr(props.device_name.as_ptr());
        log::debug!("Selected physical device: {device_name:?}");
    }

    Ok((device, graphics.unwrap(), present.unwrap()))
}

fn create_vulkan_device_and_graphics_and_present_qs(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    graphics_q_index: u32,
    present_q_index: u32,
) -> Result<(Device, vk::Queue, vk::Queue), Box<dyn Error>> {
    log::debug!("Creating vulkan device and graphics and present queues");
    let queue_priorities = [1.0f32];
    let queue_create_infos = {
        let mut indices = vec![graphics_q_index, present_q_index];
        indices.dedup();

        indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>()
    };

    let device_extensions_ptrs = [
        swapchain::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ];

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(graphics_q_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_q_index, 0) };

    Ok((device, graphics_queue, present_queue))
}

fn create_vulkan_swapchain(
    vulkan_context: &VulkanContext,
) -> Result<
    (
        swapchain::Device,
        vk::SwapchainKHR,
        vk::Extent2D,
        vk::Format,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
    ),
    Box<dyn Error>,
> {
    log::debug!("Creating vulkan swapchain");
    // Swapchain format
    let format = {
        let formats = unsafe {
            vulkan_context.surface.get_physical_device_surface_formats(
                vulkan_context.physical_device,
                vulkan_context.surface_khr,
            )?
        };

        *formats
            .iter()
            .find(|format| {
                format.format == vk::Format::R8G8B8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
    };
    log::debug!("Swapchain format: {format:?}");

    // Swapchain present mode
    let present_mode = {
        let present_modes = unsafe {
            vulkan_context
                .surface
                .get_physical_device_surface_present_modes(
                    vulkan_context.physical_device,
                    vulkan_context.surface_khr,
                )?
        };
        if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    };
    log::debug!("Swapchain present mode: {present_mode:?}");

    let capabilities = unsafe {
        vulkan_context
            .surface
            .get_physical_device_surface_capabilities(
                vulkan_context.physical_device,
                vulkan_context.surface_khr,
            )?
    };

    // Swapchain extent
    let extent = {
        if capabilities.current_extent.width != std::u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = WIDTH.min(max.width).max(min.width);
            let height = HEIGHT.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    };
    log::debug!("Swapchain extent: {extent:?}");

    // Swapchain image count
    let image_count = capabilities.min_image_count;
    log::debug!("Swapchain image count: {image_count:?}");

    // Swapchain
    let families_indices = [
        vulkan_context.graphics_q_index,
        vulkan_context.present_q_index,
    ];
    let create_info = {
        let mut builder = vk::SwapchainCreateInfoKHR::default()
            .surface(vulkan_context.surface_khr)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        builder = if vulkan_context.graphics_q_index != vulkan_context.present_q_index {
            builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families_indices)
        } else {
            builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
    };

    let swapchain = swapchain::Device::new(&vulkan_context.instance, &vulkan_context.device);
    let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None)? };

    // Swapchain images and image views
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr)? };
    let views = images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe { vulkan_context.device.create_image_view(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((
        swapchain,
        swapchain_khr,
        extent,
        format.format,
        images,
        views,
    ))
}

fn create_vulkan_render_pass(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, Box<dyn Error>> {
    log::debug!("Creating vulkan render pass");
    let attachment_descs = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color_attachment_refs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let subpass_descs = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)];

    let subpass_deps = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    Ok(unsafe { device.create_render_pass(&render_pass_info, None)? })
}

fn create_vulkan_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    image_views: &[vk::ImageView],
) -> Result<Vec<vk::Framebuffer>, Box<dyn Error>> {
    log::debug!("Creating vulkan framebuffers");
    Ok(image_views
        .iter()
        .map(|view| [*view])
        .map(|attachments| {
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&framebuffer_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?)
}

fn record_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    pixels_per_point: f32,
    renderer: &mut Renderer,

    clipped_primitives: &[ClippedPrimitive],
) -> Result<(), Box<dyn Error>> {
    unsafe { device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())? };

    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info)? };

    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        })
        .clear_values(&[vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.007, 0.007, 0.007, 1.0],
            },
        }]);

    unsafe {
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        )
    };

    renderer.cmd_draw(command_buffer, extent, pixels_per_point, clipped_primitives)?;

    unsafe { device.cmd_end_render_pass(command_buffer) };

    unsafe { device.end_command_buffer(command_buffer)? };

    Ok(())
}
