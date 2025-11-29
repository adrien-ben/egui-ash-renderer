# Changelog

## current

- new `simple-allocator` feature. enabled by default.
- the `Allocator` trait is now public.
- new `custom-allocator` feature allowing to create an renderer with a custom allocator.

### Breaking

- Renderer is now generic over the `Allocator` trait.
- Remove `dynamic-rendering` feature.
- Functions that used to take `vk::RenderPass` or `DynamicRendering` now take `RenderMode` instead.
- `Renderer::with_default_allocator` is now `Renderer::with_simple_allocator`.

## 0.12.0

### Breaking

- Remove Debug derive from DynamicRendering by @mickvangelderen
- Allow passing stencil attachment format for dynamic rendering by @mickvangelderen

## 0.11.0

- Do not wrap vk_mem::Allocator in a Mutex by @BattyBoopers

## 0.10.0

- egui 0.33
- gpu-allocator 0.28
- vk-mem 0.5

## 0.9.0

- egui 0.32
- dev: rust edition 2024

## 0.8.0

- egui >=0.26, <=0.31

## 0.7.0

- egui >=0.26, <=0.30
- thiserror 2

## 0.6.0

- egui >=0.26, <=0.29
- examples: winit 0.30

## 0.5.0

- ash 0.38
- vk-mem 0.4
- gpu-allocator 0.27

## 0.4.0

- egui >=0.26, <=0.28

## 0.3.0

- dev: bump simple_logger to 5.0
- add option to target srgb framebuffer
- allow gpu-allocator 0.26
- remove ultraviolet and fix viewport size issue
- allow updating dynamic rendering parameters

## 0.2.0

- egui >=0.26, <=0.27

## 0.1.0

- Initial implementation with default allocator and
  - support for vk-mem
  - support for gpu-allocator
  - support for dynamic rendering
