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
extern crate cubeb;
extern crate rustfft;

use cubeb::Frame;
use rustfft::{FFTplanner, FFT};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
// }}}
// {{{ General
#[macro_use]
extern crate clap;
extern crate seqlock;

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::ops::Deref;
use clap::Arg;
use seqlock::SeqLock;
// }}}

const ARG_DEVICE: &str = "DEVICE";
const ARG_FPS: &str = "FPS";
const CHANNELS: u32 = 2;
const STREAM_FORMAT: cubeb::SampleFormat = cubeb::SampleFormat::Float32NE;

type FrameType = cubeb::StereoFrame<f32>;

struct StreamCallbackImpl {
    pub fft_resolution: usize,
    pub fft: Arc<FFT<f32>>,
    pub fft_input_index: usize,
    pub fft_input: Vec<f32>,
    pub fft_input_ready: Arc<Mutex<Vec<f32>>>,
    pub consumed: Arc<SeqLock<bool>>,
}

impl StreamCallbackImpl {
    fn new(rate: u32, fps: u32) -> Self {
        let mut planner = FFTplanner::new(false);
        let fft_resolution = (rate / fps) as usize;

        StreamCallbackImpl {
            fft_resolution,
            fft: planner.plan_fft(fft_resolution),
            fft_input_index: 0,
            fft_input: vec![Zero::zero(); fft_resolution],
            fft_input_ready: Arc::new(Mutex::new(vec![Zero::zero(); fft_resolution])),
            consumed: Arc::new(SeqLock::new(false)),
        }
    }
}

impl cubeb::StreamCallback for StreamCallbackImpl {
    type Frame = FrameType;

    fn data_callback(
        &mut self,
        input: &[Self::Frame],
        _: &mut [Self::Frame]
    ) -> isize {
        for frame in input {
            let avg = (frame.l + frame.r) / 2.0;
            self.fft_input[self.fft_input_index] = avg;
            self.fft_input_index += 1;

            if self.fft_input_index >= self.fft_resolution {
                self.fft_input_index = 0;
                let mut fft_input_ready = self.fft_input_ready.deref().lock()
                    .expect("Poisoned Mutex.");

                *fft_input_ready = self.fft_input.clone();
                *self.consumed.lock_write() = false;
            }
        }

        input.len() as isize
    }

    fn state_callback(&mut self, state: cubeb::State) {
        println!("Stream state: {:?}", state);
    }
}

fn main() {
    let matches = app_from_crate!()
        .arg(Arg::with_name(ARG_FPS)
             .long("fps")
             .short("f")
             .help("The framerate to calculate the audio FFT at."))
        .arg(Arg::with_name(ARG_DEVICE))
        .get_matches();
    let requested_device_index: Option<usize> = if let Some(arg) = matches.value_of(ARG_DEVICE) {
        match arg.parse() {
            Ok(arg) => Some(arg),
            Err(_) => None,
        }
    } else {
        None
    };
    let fps: u32 = matches.value_of(ARG_FPS).and_then(|fps| {
        match fps.parse() {
            Ok(fps) => Some(fps),
            Err(_) => None,
        }
    }).unwrap_or(60);
    let ctx = cubeb::Context::init("streamsplash-cubeb-ctx", None)
        .expect("Could not create a Cubeb context.");
    let devices = ctx.enumerate_devices(cubeb::DEVICE_TYPE_INPUT)
        .expect("Could not enumerate Cubeb devices.");

    if let None = requested_device_index {
        println!("Choose one of the following devices by their index:");

        let mut device_index = 0;

        for device_info in (*devices).iter() {
            if !device_info.friendly_name().is_some() {
                continue;
            }

            println!("{}: {}", device_index, device_info.friendly_name().unwrap());
            device_index += 1;
        }

        std::process::exit(0);
    }

    let selected_device_info = (*devices).iter().nth(requested_device_index.unwrap());

    if let None = selected_device_info {
        println!("No such device found.");
        std::process::exit(1);
    }

    let selected_device_info = selected_device_info.unwrap();

    println!("Selected device: {}", selected_device_info.friendly_name().unwrap());

    let rate = selected_device_info.default_rate();
    let params = cubeb::StreamParamsBuilder::new()
        .format(STREAM_FORMAT)
        .rate(rate)
        .channels(CHANNELS)
        .layout(FrameType::layout())
        .take();
    let latency = match selected_device_info.latency_lo() {
        0 => ctx.min_latency(&params).expect("Could not get min latency"),
        l => l,
    };
    let stream_init_opts = cubeb::StreamInitOptionsBuilder::new()
        .stream_name("streamsplash-cubeb-stream")
        .input_device(selected_device_info.devid())
        .input_stream_param(&params)
        .latency(latency)
        .take();
    let callback = StreamCallbackImpl::new(rate, fps);
    let fft_resolution = callback.fft_resolution;
    let fft = callback.fft.clone();
    let fft_input_ready = callback.fft_input_ready.clone();
    let consumed = callback.consumed.clone();
    let stream = ctx.stream_init(
        &stream_init_opts,
        callback,
    ).map_err(|e| panic!("Failed to create Cubeb stream: {:?}", e)).unwrap();

    stream.start().unwrap();

    {
        std::thread::spawn(move || {
            let mut fft_input_vec = Vec::with_capacity(fft_resolution);
            let mut fft_output_vec = Vec::with_capacity(fft_resolution);

            loop {
                if !consumed.read() {
                    let mut consumed = consumed.lock_write();
                    let fft_input_ready = fft_input_ready.lock().unwrap();

                    for ref float in fft_input_ready.deref() {
                        fft_input_vec.push(Complex::new(**float, 0_f32));
                    }

                    fft_output_vec.resize(fft_input_vec.len(), Complex::new(0_f32, 0_f32));
                    fft.process(&mut fft_input_vec, &mut fft_output_vec);
                    fft_input_vec.clear();
                    fft_output_vec.clear();

                    *consumed = true;
                }

                std::thread::sleep(Duration::from_millis(1));
            }
        });
    }

    loop {}
    // std::thread::sleep(Duration::from_millis(500));
    // stream.stop().unwrap();
}

// END OF AUDIO CODE
// BEGINNING OF GPU CODE

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
