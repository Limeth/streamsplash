// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// For the purpose of this example all unused code is allowed.
#![allow(dead_code)]

// {{{ Graphics
extern crate cgmath;
extern crate image;
extern crate winit;

#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;
use vulkano::sync::GpuFuture;
// }}}
// {{{ Audio
extern crate portaudio;
// }}}
// {{{ General
use std::thread;
use std::sync::Arc;
use std::time::Duration;
// }}}

const CHANNELS: i32 = 2;
const NUM_SECONDS: i32 = 5;
// const SAMPLE_RATE: f64 = 44_100.0;
const FRAMES_PER_BUFFER: u32 = 64;

fn main() {
    let requested_device_index = Some(8);  // TODO: Make an argument for it
    let pa = portaudio::PortAudio::new().expect("Could not initialize PortAudio.");
    
    if let None = requested_device_index {
        let devices: portaudio::Devices = pa.devices().expect("Couldn't request audio devices.");

        println!("Choose one of the following devices by their index:");

        for device in devices {
            match device {
                Ok((device_index, device_info)) => {
                    println!("{}: {}", device_index.0, device_info.name);
                },
                Err(error) => {
                    panic!("Could not retrieve the next device: {}", error);
                }
            }
        }

        std::process::exit(0);
    }

    let device_index = portaudio::DeviceIndex(requested_device_index.unwrap());
    let device_info = pa.device_info(device_index).expect("Could not request info about the desired device.");
    let settings = portaudio::stream::InputSettings::with_flags(
        portaudio::stream::Parameters::new(
            device_index,
            CHANNELS,
            true,
            device_info.default_low_input_latency
        ),
        device_info.default_sample_rate,
        FRAMES_PER_BUFFER,
        portaudio::stream::flags::Flags::empty()
    );

    // This routine will be called by the PortAudio engine when audio is needed. It may called at
    // interrupt level on some machines so don't do anything that could mess up the system like
    // dynamic resource allocation or IO.
    let callback = move |portaudio::InputStreamCallbackArgs::<i32> { buffer, frames, flags, time }| {
        for idx in (0..frames).map(|x| 2 * x) {
            println!("{} : {}", buffer[idx], buffer[idx + 1]);
        }
        portaudio::Continue
    };

    let mut stream = pa.open_non_blocking_stream(settings, callback)
        .expect("Could not open a non-blocking stream.");

    stream.start().expect("Could not start a non-blocking stream.");

    loop {}
}

fn main_t() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, &[]).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let queue = physical.queue_families().find(|&q| q.supports_graphics() &&
                                                   window.surface().is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };
    let (device, mut queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                            &device_ext, [(queue, 0.5)].iter().cloned())
                               .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let caps = window.surface().capabilities(physical).expect("failed to get surface capabilities");

        let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
        let usage = caps.supported_usage_flags;

        vulkano::swapchain::Swapchain::new(device.clone(), window.surface().clone(), caps.min_image_count,
                                           vulkano::format::B8G8R8A8Srgb, dimensions, 1,
                                           usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           vulkano::swapchain::CompositeAlpha::Opaque,
                                           vulkano::swapchain::PresentMode::Fifo, true, None).expect("failed to create swapchain")
    };


    #[derive(Debug, Clone)]
    struct Vertex { position: [f32; 2] }
    impl_vertex!(Vertex, position);

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex]>
                               ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(),
                                       Some(queue.family()), [
                                           Vertex { position: [-0.5, -0.5 ] },
                                           Vertex { position: [-0.5,  0.5 ] },
                                           Vertex { position: [ 0.5, -0.5 ] },
                                           Vertex { position: [ 0.5,  0.5 ] },
                                       ].iter().cloned()).expect("failed to create buffer");

    let vs = vs::Shader::load(&device).expect("failed to create shader module");
    let fs = fs::Shader::load(&device).expect("failed to create shader module");

    let renderpass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()
    );

    let texture = vulkano::image::immutable::ImmutableImage::new(device.clone(), vulkano::image::Dimensions::Dim2d { width: 93, height: 93 },
                                                                 vulkano::format::R8G8B8A8Unorm, Some(queue.family())).unwrap();


    let pixel_buffer = {
        let image = image::load_from_memory_with_format(include_bytes!("image_img.png"),
                                                        image::ImageFormat::PNG).unwrap().to_rgba();
        let image_data = image.into_raw().clone();

        let image_data_chunks = image_data.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]);

        // TODO: staging buffer instead
        vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[[u8; 4]]>
            ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(),
                        Some(queue.family()), image_data_chunks)
                        .expect("failed to create buffer")
    };


    let sampler = vulkano::sampler::Sampler::new(device.clone(), vulkano::sampler::Filter::Linear,
                                                 vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 0.0, 1.0, 0.0, 0.0).unwrap();

    let pipeline = Arc::new(vulkano::pipeline::GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_strip()
        .viewports(std::iter::once(vulkano::pipeline::viewport::Viewport {
            origin: [0.0, 0.0],
            depth_range: 0.0 .. 1.0,
            dimensions: [images[0].dimensions()[0] as f32, images[0].dimensions()[1] as f32],
        }))
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let set = Arc::new(simple_descriptor_set!(pipeline.clone(), 0, {
        tex: (texture.clone(), sampler.clone())
    }));

    let framebuffers = images.iter().map(|image| {
        Arc::new(vulkano::framebuffer::Framebuffer::start(renderpass.clone())
            .add(image.clone()).unwrap().build().unwrap())
    }).collect::<Vec<_>>();

    let mut previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        let (image_num, future) = vulkano::swapchain::acquire_next_image(swapchain.clone(), None).unwrap();

        let cb = vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .copy_buffer_to_image(pixel_buffer.clone(), texture.clone())
            .unwrap()
            //.clear_color_image(&texture, [0.0, 1.0, 0.0, 1.0])
            .begin_render_pass(
                framebuffers[image_num].clone(), false,
                vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), vulkano::command_buffer::DynamicState::none(), vertex_buffer.clone(),
                  set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(future)
            .then_execute(queue.clone(), cb).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        previous_frame_end = Box::new(future) as Box<_>;

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = texture(tex, tex_coords);
}
"]
    struct Dummy;
}
