use egui::Context;
use egui_wgpu::Renderer;
use egui_winit::State;
use winit::window::Window;

#[derive(Debug)]
pub enum MenuAction {
    OpenSkin,
    SavePreset,
    LoadPreset,
    Quit,
    ToggleFullscreen,
    ToggleCheatSheet,
    ToggleRecording,
    TogglePanels,
}

#[derive(Debug)]
pub enum ParamChange {
    FoldCount(f32),
    Zoom(f32),
    RotationSpeedScale(f32),
    FrameSize(f32),
    FrameColorHue(f32),
    InvertEnabled(bool),
    ColorizeEnabled(bool),
    ColorizeHue(f32),
    ColorizeIntensity(f32),
    DistortionEnabled(bool),
    DistortionAmplitude(f32),
    DistortionFrequency(f32),
    ShakeEnabled(bool),
    BassZoomStrength(f32),
    CurrentShape(crate::ShapeKind),
    FrameShape(crate::FrameShape),
    PainterKind(crate::PainterKind),
}

pub struct MenuBar {
    pub ctx: Context,
    state: State,
    renderer: Renderer,
    pub pending_actions: Vec<MenuAction>,
    pub pending_param_changes: Vec<ParamChange>,
    pub params_panel_visible: bool,
}

impl MenuBar {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        let ctx = Context::default();
        let state = State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = Renderer::new(device, surface_format, None, 1, false);
        Self {
            ctx,
            state,
            renderer,
            pending_actions: Vec::new(),
            pending_param_changes: Vec::new(),
            params_panel_visible: true,
        }
    }

    #[allow(dead_code)]
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    #[allow(dead_code)]
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.state.on_window_event(window, event);
        match event {
            winit::event::WindowEvent::KeyboardInput { .. } => {
                self.ctx.wants_keyboard_input()
            }
            winit::event::WindowEvent::MouseInput { .. }
            | winit::event::WindowEvent::CursorMoved { .. }
            | winit::event::WindowEvent::MouseWheel { .. } => {
                response.consumed && self.ctx.wants_pointer_input()
            }
            _ => response.consumed,
        }
    }

    pub fn take_actions(&mut self) -> Vec<MenuAction> {
        std::mem::take(&mut self.pending_actions)
    }

    pub fn take_param_changes(&mut self) -> Vec<ParamChange> {
        std::mem::take(&mut self.pending_param_changes)
    }

    pub fn toggle_params_panel(&mut self) {
        self.params_panel_visible = !self.params_panel_visible;
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        screen_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        current_params: &crate::VisualParams,
    ) {
        let raw_input = self.state.take_egui_input(window);

        // Capture fields by copy/value before the closure to avoid borrowing self inside it.
        let panel_visible = self.params_panel_visible;
        let mut frame_actions: Vec<MenuAction> = Vec::new();
        let mut frame_changes: Vec<ParamChange> = Vec::new();

        let full_output = self.ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open Skin...").clicked() {
                            frame_actions.push(MenuAction::OpenSkin);
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Save Preset  Ctrl+S").clicked() {
                            frame_actions.push(MenuAction::SavePreset);
                            ui.close_menu();
                        }
                        if ui.button("Load Preset  Ctrl+L").clicked() {
                            frame_actions.push(MenuAction::LoadPreset);
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            frame_actions.push(MenuAction::Quit);
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        ui.label("(Coming in 24d)");
                    });

                    ui.menu_button("View", |ui| {
                        if ui.button("Fullscreen  F11").clicked() {
                            frame_actions.push(MenuAction::ToggleFullscreen);
                            ui.close_menu();
                        }
                        if ui.button("Show Cheat Sheet  ?").clicked() {
                            frame_actions.push(MenuAction::ToggleCheatSheet);
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Tools", |ui| {
                        if ui.button("Toggle Recording  F12").clicked() {
                            frame_actions.push(MenuAction::ToggleRecording);
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Window", |ui| {
                        let mut vis = panel_visible;
                        if ui.checkbox(&mut vis, "Show Parameters Panel  M").changed() {
                            frame_actions.push(MenuAction::TogglePanels);
                        }
                    });

                    ui.menu_button("Help", |ui| {
                        ui.label("abstrakt-deck v0.1.0");
                        ui.separator();
                        if ui.button("Show Cheat Sheet  ?").clicked() {
                            frame_actions.push(MenuAction::ToggleCheatSheet);
                            ui.close_menu();
                        }
                    });
                });
            });

            if panel_visible {
                egui::SidePanel::right("params_panel")
                    .resizable(true)
                    .default_width(280.0)
                    .min_width(200.0)
                    .show(ctx, |ui| {
                        ui.heading("Visualizer");
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            Self::geometry_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::frame_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::effects_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::audio_section(ui, current_params, &mut frame_changes);
                        });
                    });
            }
        });

        self.pending_actions.append(&mut frame_actions);
        self.pending_param_changes.append(&mut frame_changes);

        self.state.handle_platform_output(window, full_output.platform_output);

        let tris = self.ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: full_output.pixels_per_point,
        };

        self.renderer.update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            self.renderer.render(&mut render_pass, &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    fn geometry_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Geometry", |ui| {
            let mut shape = params.current_shape;
            egui::ComboBox::from_label("Shape")
                .selected_text(shape.name())
                .show_ui(ui, |ui| {
                    for v in [
                        crate::ShapeKind::Cylinder,
                        crate::ShapeKind::Sphere,
                        crate::ShapeKind::Cube,
                        crate::ShapeKind::Tetrahedron,
                    ] {
                        ui.selectable_value(&mut shape, v, v.name());
                    }
                });
            if shape != params.current_shape {
                changes.push(ParamChange::CurrentShape(shape));
            }

            let mut painter = params.painter_kind;
            egui::ComboBox::from_label("Painter")
                .selected_text(painter.name())
                .show_ui(ui, |ui| {
                    for v in [
                        crate::PainterKind::HueStripe,
                        crate::PainterKind::Spiral,
                        crate::PainterKind::Plasma,
                        crate::PainterKind::Skin,
                    ] {
                        ui.selectable_value(&mut painter, v, v.name());
                    }
                });
            if painter != params.painter_kind {
                changes.push(ParamChange::PainterKind(painter));
            }

            let mut fold = params.fold_count;
            if ui
                .add(egui::Slider::new(&mut fold, 2.0..=24.0).text("Fold Count").integer())
                .changed()
            {
                changes.push(ParamChange::FoldCount(fold));
            }

            let mut zoom = params.zoom;
            if ui
                .add(egui::Slider::new(&mut zoom, 0.3..=1.5).text("Zoom").step_by(0.05))
                .changed()
            {
                changes.push(ParamChange::Zoom(zoom));
            }

            let mut rot = params.rotation_speed_scale;
            if ui
                .add(
                    egui::Slider::new(&mut rot, 0.0..=4.0)
                        .text("Rotation Speed")
                        .step_by(0.25),
                )
                .changed()
            {
                changes.push(ParamChange::RotationSpeedScale(rot));
            }
        });
    }

    fn frame_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Frame", |ui| {
            let mut fs = params.frame_shape;
            egui::ComboBox::from_label("Frame Shape")
                .selected_text(format!("{:?}", fs))
                .show_ui(ui, |ui| {
                    for v in [
                        crate::FrameShape::None,
                        crate::FrameShape::Circle,
                        crate::FrameShape::Square,
                        crate::FrameShape::Rounded,
                        crate::FrameShape::Hexagon,
                        crate::FrameShape::Octagon,
                        crate::FrameShape::Star,
                    ] {
                        ui.selectable_value(&mut fs, v, format!("{:?}", v));
                    }
                });
            if fs != params.frame_shape {
                changes.push(ParamChange::FrameShape(fs));
            }

            let mut size = params.frame_size;
            if ui
                .add(egui::Slider::new(&mut size, 0.4..=1.0).text("Size").step_by(0.05))
                .changed()
            {
                changes.push(ParamChange::FrameSize(size));
            }

            let mut hue = params.frame_color_hue;
            if ui
                .add(
                    egui::Slider::new(&mut hue, 0.0..=360.0)
                        .text("Color Hue")
                        .step_by(5.0)
                        .suffix("°"),
                )
                .changed()
            {
                changes.push(ParamChange::FrameColorHue(hue));
            }
        });
    }

    fn effects_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Effects", |ui| {
            let mut invert = params.invert_enabled;
            if ui.checkbox(&mut invert, "Color Invert").changed() {
                changes.push(ParamChange::InvertEnabled(invert));
            }

            ui.separator();

            let mut colorize_on = params.colorize_enabled;
            if ui.checkbox(&mut colorize_on, "Colorize Tint").changed() {
                changes.push(ParamChange::ColorizeEnabled(colorize_on));
            }
            ui.add_enabled_ui(colorize_on, |ui| {
                let mut hue = params.colorize_hue;
                if ui
                    .add(
                        egui::Slider::new(&mut hue, 0.0..=360.0)
                            .text("Hue")
                            .step_by(5.0)
                            .suffix("°"),
                    )
                    .changed()
                {
                    changes.push(ParamChange::ColorizeHue(hue));
                }
                let mut intensity = params.colorize_intensity;
                if ui
                    .add(
                        egui::Slider::new(&mut intensity, 0.0..=1.0)
                            .text("Intensity")
                            .step_by(0.05),
                    )
                    .changed()
                {
                    changes.push(ParamChange::ColorizeIntensity(intensity));
                }
            });

            ui.separator();

            let mut dist_on = params.distortion_enabled;
            if ui.checkbox(&mut dist_on, "Distortion").changed() {
                changes.push(ParamChange::DistortionEnabled(dist_on));
            }
            ui.add_enabled_ui(dist_on, |ui| {
                let mut amp = params.distortion_amplitude;
                if ui
                    .add(
                        egui::Slider::new(&mut amp, 0.0..=0.5)
                            .text("Amplitude")
                            .step_by(0.02),
                    )
                    .changed()
                {
                    changes.push(ParamChange::DistortionAmplitude(amp));
                }
                let mut freq = params.distortion_frequency;
                if ui
                    .add(
                        egui::Slider::new(&mut freq, 0.5..=8.0)
                            .text("Frequency")
                            .step_by(0.5),
                    )
                    .changed()
                {
                    changes.push(ParamChange::DistortionFrequency(freq));
                }
            });
        });
    }

    fn audio_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Audio", |ui| {
            let mut shake = params.shake_enabled;
            if ui.checkbox(&mut shake, "Beat-reactive Shake").changed() {
                changes.push(ParamChange::ShakeEnabled(shake));
            }

            let mut bass = params.bass_zoom_strength;
            if ui
                .add(
                    egui::Slider::new(&mut bass, 0.0..=1.0)
                        .text("Bass-Zoom Strength")
                        .step_by(0.05),
                )
                .changed()
            {
                changes.push(ParamChange::BassZoomStrength(bass));
            }
        });
    }
}
