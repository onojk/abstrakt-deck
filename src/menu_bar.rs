use egui::Context;
use egui_wgpu::Renderer;
use egui_winit::State;
use winit::window::Window;

#[derive(Debug, Clone, Copy)]
pub enum LockTarget {
    PainterKind,
    CurrentShape,
    FoldCount,
    Zoom,
    RotationSpeedScale,
    FrameShape,
    FrameSize,
    FrameColorHue,
    InvertEnabled,
    ColorizeEnabled,
    ColorizeHue,
    ColorizeIntensity,
    DistortionEnabled,
    DistortionAmplitude,
    DistortionFrequency,
    DistortionPlusEnabled,
    DistortionPlusYaw,
    DistortionPlusPitch,
    DistortionPlusRoll,
    Contrast,
    ContrastPasses,
    Saturation,
    BassZoomStrength,
    BeatReactivity,
    MidiShakeEnabled,
    AudioShakeEnabled,
    RibbonsEnabled,
    RibbonsIntensity,
    PaletteMode,
    PaletteTint,
    PaletteMonoHue,
    BlackholeEnabled,
    BlackholeWarpStrength,
    BlackholeWarpCurve,
    BlackholeAlphaRadius,
    BlackholeWanderAmount,
    ColorHarmony,
    ColorAnchorHue,
    ColorSaturation,
    ColorValue,
    PhantomEnabled,
    PhantomDelaySeconds,
    PhantomKeyColor,
    PhantomKeyTolerance,
    PhantomKeySoftness,
    PhantomKeyStrength,
    PhantomOpacity,
}

#[derive(Debug, Clone)]
pub struct PlayerInfo {
    pub filename:         String,
    pub duration_seconds: f32,
    pub position_seconds: f32,
    pub is_playing:       bool,
}

#[derive(Debug)]
pub enum MenuAction {
    OpenSkin,
    OpenAudio,
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
    DistortionPlusEnabled(bool),
    DistortionPlusYaw(f32),
    DistortionPlusPitch(f32),
    DistortionPlusRoll(f32),
    MidiShakeEnabled(bool),
    AudioShakeEnabled(bool),
    RibbonsEnabled(bool),
    RibbonsIntensity(f32),
    BassZoomStrength(f32),
    BeatReactivity(f32),
    CurrentShape(crate::ShapeKind),
    FrameShape(crate::FrameShape),
    PainterKind(crate::PainterKind),
    SkinCropOffset(f32),
    Contrast(f32),
    Saturation(f32),
    ContrastPasses(u32),
    RandomModeEnabled(bool),
    RandomModeAggressiveness(f32),
    ReactiveModeEnabled(bool),
    ReactiveModeAggressiveness(f32),
    PartyModeEnabled(bool),
    PartyModeAggressiveness(f32),
    ToggleLock(LockTarget),
    PlayerToggle,
    PlayerStop,
    PlayerSeek(f32),
    ExportResolution(crate::ResolutionPreset),
    ExportFramerate(crate::FramerateChoice),
    SetExportLivePreview(bool),
    TriggerExport,
    SetAudioSourceMode(crate::AudioSourceMode),
    SetPaletteMode(crate::PaletteMode),
    PaletteTint(f32),
    PaletteMonoHue(f32),
    BlackholeEnabled(bool),
    BlackholeWarpStrength(f32),
    BlackholeWarpCurve(f32),
    BlackholeAlphaRadius(f32),
    BlackholeWanderAmount(f32),
    ColorHarmony(crate::color::ColorHarmony),
    ColorAnchorHue(f32),
    ColorSaturation(f32),
    ColorValue(f32),
    PhantomEnabled(bool),
    PhantomDelaySeconds(f32),
    PhantomKeyColor([f32; 3]),
    PhantomKeyTolerance(f32),
    PhantomKeySoftness(f32),
    PhantomKeyStrength(f32),
    PhantomOpacity(f32),
}

#[derive(Clone, Copy)]
pub struct ExportProgress {
    pub current_frame: u32,
    pub total_frames: u32,
    pub is_muxing: bool,
}

pub struct MenuBar {
    pub ctx: Context,
    state: State,
    renderer: Renderer,
    pub pending_actions: Vec<MenuAction>,
    pub pending_param_changes: Vec<ParamChange>,
    pub params_panel_visible: bool,
    pub skin_thumbnail: Option<egui::TextureHandle>,
    skin_thumbnail_aspect: f32,
    pub current_crop_y_offset: f32,
    pub player_info: Option<PlayerInfo>,
    pub export_progress: Option<ExportProgress>,
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
            skin_thumbnail: None,
            skin_thumbnail_aspect: 1.0,
            current_crop_y_offset: 0.5,
            player_info: None,
            export_progress: None,
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
                // Only block our hotkeys if egui actually consumed this key
                // (e.g. text typed into a text-edit widget). wants_keyboard_input()
                // returns true whenever any widget has focus (sliders too) and
                // would silently swallow B/N/Y/Space even with no text field active.
                response.consumed
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

    pub fn set_skin_thumbnail(&mut self, source_img: &image::DynamicImage) {
        use image::GenericImageView;
        const MAX_W: u32 = 280;
        const MAX_H: u32 = 200;
        let (w, h) = source_img.dimensions();
        self.skin_thumbnail_aspect = w as f32 / h as f32;
        let scale = (MAX_W as f32 / w as f32).min(MAX_H as f32 / h as f32);
        let tw = ((w as f32 * scale) as u32).max(1);
        let th = ((h as f32 * scale) as u32).max(1);
        let resized = source_img.resize_exact(tw, th, image::imageops::FilterType::Triangle);
        let pixels = resized.to_rgba8().into_raw();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [tw as usize, th as usize],
            &pixels,
        );
        self.skin_thumbnail = Some(self.ctx.load_texture(
            "skin_thumbnail",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
    }

    #[allow(dead_code)]
    pub fn clear_skin_thumbnail(&mut self) {
        self.skin_thumbnail = None;
    }

    #[allow(clippy::too_many_arguments)]
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
        let skin_thumb = self.skin_thumbnail.clone();
        let skin_aspect = self.skin_thumbnail_aspect;
        let mut current_crop = self.current_crop_y_offset;
        let player_info_snap = self.player_info.clone();
        let export_progress_snap = self.export_progress;
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
                        if ui.button("Open Audio...").clicked() {
                            frame_actions.push(MenuAction::OpenAudio);
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
                            Self::distortion_plus_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::audio_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::skin_section_static(
                                ui,
                                skin_thumb.as_ref(),
                                skin_aspect,
                                &mut current_crop,
                                &mut frame_changes,
                            );
                            ui.separator();
                            Self::audio_player_section(
                                ui,
                                player_info_snap.as_ref(),
                                &mut frame_changes,
                            );
                            ui.separator();
                            Self::export_section(
                                ui,
                                current_params,
                                player_info_snap.as_ref(),
                                export_progress_snap,
                                &mut frame_changes,
                            );
                            ui.separator();
                            Self::modes_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::ribbons_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::palette_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::blackhole_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::phantom_section(ui, current_params, &mut frame_changes);
                            ui.separator();
                            Self::color_theory_section(ui, current_params, &mut frame_changes);
                        });
                    });
            }
        });

        self.current_crop_y_offset = current_crop;
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

    fn lock_button(ui: &mut egui::Ui, locked: bool) -> egui::Response {
        let icon = if locked { "🔒" } else { "🔓" };
        let button = egui::Button::new(icon).small();
        let button = if locked {
            button.fill(egui::Color32::from_rgb(80, 60, 30))
        } else {
            button
        };
        ui.add(button)
    }

    fn geometry_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Geometry", |ui| {
            // Shape
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.current_shape).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::CurrentShape));
                }
                ui.add_enabled_ui(!params.locks.current_shape, |ui| {
                    let mut shape = params.current_shape;
                    egui::ComboBox::from_label("Shape")
                        .selected_text(shape.name())
                        .show_ui(ui, |ui| {
                            for v in [
                                crate::ShapeKind::Cylinder,
                                crate::ShapeKind::Sphere,
                                crate::ShapeKind::Cube,
                                crate::ShapeKind::Tetrahedron,
                                crate::ShapeKind::Icosahedron,
                                crate::ShapeKind::Urchin,
                                crate::ShapeKind::Caltrop,
                            ] {
                                ui.selectable_value(&mut shape, v, v.name());
                            }
                        });
                    if shape != params.current_shape {
                        changes.push(ParamChange::CurrentShape(shape));
                    }
                });
            });

            // Painter
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.painter_kind).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::PainterKind));
                }
                ui.add_enabled_ui(!params.locks.painter_kind, |ui| {
                    let mut painter = params.painter_kind;
                    egui::ComboBox::from_label("Painter")
                        .selected_text(painter.name())
                        .show_ui(ui, |ui| {
                            for v in [
                                crate::PainterKind::HueStripe,
                                crate::PainterKind::Spiral,
                                crate::PainterKind::Plasma,
                                crate::PainterKind::Skin,
                                crate::PainterKind::AudioPaint,
                                crate::PainterKind::PrintHead,
                                crate::PainterKind::Image,
                            ] {
                                ui.selectable_value(&mut painter, v, v.name());
                            }
                        });
                    if painter != params.painter_kind {
                        changes.push(ParamChange::PainterKind(painter));
                    }
                });
            });

            // Fold Count
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.fold_count).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::FoldCount));
                }
                ui.add_enabled_ui(!params.locks.fold_count, |ui| {
                    let mut fold = params.fold_count;
                    if ui.add(egui::Slider::new(&mut fold, 2.0..=24.0).text("Fold Count").integer()).changed() {
                        changes.push(ParamChange::FoldCount(fold));
                    }
                });
            });

            // Zoom
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.zoom).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::Zoom));
                }
                ui.add_enabled_ui(!params.locks.zoom, |ui| {
                    let mut zoom = params.zoom;
                    if ui.add(egui::Slider::new(&mut zoom, 0.3..=1.5).text("Zoom").step_by(0.05)).changed() {
                        changes.push(ParamChange::Zoom(zoom));
                    }
                });
            });

            // Rotation Speed
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.rotation_speed_scale).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::RotationSpeedScale));
                }
                ui.add_enabled_ui(!params.locks.rotation_speed_scale, |ui| {
                    let mut rot = params.rotation_speed_scale;
                    if ui.add(egui::Slider::new(&mut rot, 0.0..=4.0).text("Rotation Speed").step_by(0.25)).changed() {
                        changes.push(ParamChange::RotationSpeedScale(rot));
                    }
                });
            });
        });
    }

    fn frame_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Frame", |ui| {
            // Frame Shape
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.frame_shape).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::FrameShape));
                }
                ui.add_enabled_ui(!params.locks.frame_shape, |ui| {
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
                                crate::FrameShape::Flower,
                                crate::FrameShape::Star,
                            ] {
                                ui.selectable_value(&mut fs, v, format!("{:?}", v));
                            }
                        });
                    if fs != params.frame_shape {
                        changes.push(ParamChange::FrameShape(fs));
                    }
                });
            });

            // Size
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.frame_size).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::FrameSize));
                }
                ui.add_enabled_ui(!params.locks.frame_size, |ui| {
                    let mut size = params.frame_size;
                    if ui.add(egui::Slider::new(&mut size, 0.4..=1.0).text("Size").step_by(0.05)).changed() {
                        changes.push(ParamChange::FrameSize(size));
                    }
                });
            });

            // Color Hue
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.frame_color_hue).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::FrameColorHue));
                }
                ui.add_enabled_ui(!params.locks.frame_color_hue, |ui| {
                    let mut hue = params.frame_color_hue;
                    if ui.add(egui::Slider::new(&mut hue, 0.0..=360.0).text("Color Hue").step_by(5.0).suffix("°")).changed() {
                        changes.push(ParamChange::FrameColorHue(hue));
                    }
                });
            });
        });
    }

    fn effects_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Effects", |ui| {
            // Contrast
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.contrast).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::Contrast));
                }
                ui.add_enabled_ui(!params.locks.contrast, |ui| {
                    let mut contrast = params.contrast;
                    if ui.add(egui::Slider::new(&mut contrast, 0.0..=2.0).text("Contrast").step_by(0.05)).changed() {
                        changes.push(ParamChange::Contrast(contrast));
                    }
                });
            });

            // Contrast Passes
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.contrast_passes).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ContrastPasses));
                }
                ui.add_enabled_ui(!params.locks.contrast_passes, |ui| {
                    let mut passes = params.contrast_passes;
                    if ui.add(egui::Slider::new(&mut passes, 1..=6).text("Contrast Passes")).changed() {
                        changes.push(ParamChange::ContrastPasses(passes));
                    }
                });
            });

            // Saturation
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.saturation).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::Saturation));
                }
                ui.add_enabled_ui(!params.locks.saturation, |ui| {
                    let mut saturation = params.saturation;
                    if ui.add(egui::Slider::new(&mut saturation, 0.0..=2.0).text("Saturation").step_by(0.05)).changed() {
                        changes.push(ParamChange::Saturation(saturation));
                    }
                });
            });

            ui.separator();

            // Color Invert
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.invert_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::InvertEnabled));
                }
                ui.add_enabled_ui(!params.locks.invert_enabled, |ui| {
                    let mut invert = params.invert_enabled;
                    if ui.checkbox(&mut invert, "Color Invert").changed() {
                        changes.push(ParamChange::InvertEnabled(invert));
                    }
                });
            });

            ui.separator();

            // Colorize Tint toggle
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.colorize_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorizeEnabled));
                }
                ui.add_enabled_ui(!params.locks.colorize_enabled, |ui| {
                    let mut colorize_on = params.colorize_enabled;
                    if ui.checkbox(&mut colorize_on, "Colorize Tint").changed() {
                        changes.push(ParamChange::ColorizeEnabled(colorize_on));
                    }
                });
            });

            let colorize_active = params.colorize_enabled;

            // Colorize Hue — greyed if locked OR if colorize is off
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.colorize_hue).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorizeHue));
                }
                ui.add_enabled_ui(!params.locks.colorize_hue && colorize_active, |ui| {
                    let mut hue = params.colorize_hue;
                    if ui.add(egui::Slider::new(&mut hue, 0.0..=360.0).text("Hue").step_by(5.0).suffix("°")).changed() {
                        changes.push(ParamChange::ColorizeHue(hue));
                    }
                });
            });

            // Colorize Intensity
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.colorize_intensity).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorizeIntensity));
                }
                ui.add_enabled_ui(!params.locks.colorize_intensity && colorize_active, |ui| {
                    let mut intensity = params.colorize_intensity;
                    if ui.add(egui::Slider::new(&mut intensity, 0.0..=1.0).text("Intensity").step_by(0.05)).changed() {
                        changes.push(ParamChange::ColorizeIntensity(intensity));
                    }
                });
            });

            ui.separator();

            // Distortion toggle
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionEnabled));
                }
                ui.add_enabled_ui(!params.locks.distortion_enabled, |ui| {
                    let mut dist_on = params.distortion_enabled;
                    if ui.checkbox(&mut dist_on, "Distortion").changed() {
                        changes.push(ParamChange::DistortionEnabled(dist_on));
                    }
                });
            });

            let dist_active = params.distortion_enabled;

            // Distortion Amplitude
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_amplitude).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionAmplitude));
                }
                ui.add_enabled_ui(!params.locks.distortion_amplitude && dist_active, |ui| {
                    let mut amp = params.distortion_amplitude;
                    if ui.add(egui::Slider::new(&mut amp, 0.0..=0.5).text("Amplitude").step_by(0.02)).changed() {
                        changes.push(ParamChange::DistortionAmplitude(amp));
                    }
                });
            });

            // Distortion Frequency
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_frequency).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionFrequency));
                }
                ui.add_enabled_ui(!params.locks.distortion_frequency && dist_active, |ui| {
                    let mut freq = params.distortion_frequency;
                    if ui.add(egui::Slider::new(&mut freq, 0.5..=8.0).text("Frequency").step_by(0.5)).changed() {
                        changes.push(ParamChange::DistortionFrequency(freq));
                    }
                });
            });
        });
    }

    fn distortion_plus_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Distortion Plus", |ui| {
            // Enabled toggle
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_plus_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionPlusEnabled));
                }
                ui.add_enabled_ui(!params.locks.distortion_plus_enabled, |ui| {
                    let mut en = params.distortion_plus_enabled;
                    if ui.checkbox(&mut en, "Enabled").changed() {
                        changes.push(ParamChange::DistortionPlusEnabled(en));
                    }
                });
            });

            // Yaw slider
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_plus_yaw).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionPlusYaw));
                }
                ui.add_enabled_ui(!params.locks.distortion_plus_yaw, |ui| {
                    let mut yaw = params.distortion_plus_yaw;
                    if ui.add(egui::Slider::new(&mut yaw, -180.0..=180.0).text("Yaw").suffix("°").step_by(1.0)).changed() {
                        changes.push(ParamChange::DistortionPlusYaw(yaw));
                    }
                });
            });

            // Pitch slider
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_plus_pitch).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionPlusPitch));
                }
                ui.add_enabled_ui(!params.locks.distortion_plus_pitch, |ui| {
                    let mut pitch = params.distortion_plus_pitch;
                    if ui.add(egui::Slider::new(&mut pitch, -90.0..=90.0).text("Pitch").suffix("°").step_by(1.0)).changed() {
                        changes.push(ParamChange::DistortionPlusPitch(pitch));
                    }
                });
            });

            // Roll slider
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.distortion_plus_roll).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::DistortionPlusRoll));
                }
                ui.add_enabled_ui(!params.locks.distortion_plus_roll, |ui| {
                    let mut roll = params.distortion_plus_roll;
                    if ui.add(egui::Slider::new(&mut roll, -180.0..=180.0).text("Roll").suffix("°").step_by(1.0)).changed() {
                        changes.push(ParamChange::DistortionPlusRoll(roll));
                    }
                });
            });

            // Reset button — zeroes angles only, does not toggle enabled
            if ui.button("Reset angles").clicked() {
                changes.push(ParamChange::DistortionPlusYaw(0.0));
                changes.push(ParamChange::DistortionPlusPitch(0.0));
                changes.push(ParamChange::DistortionPlusRoll(0.0));
            }
        });
    }

    fn audio_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Audio", |ui| {
            // Source mode selector
            ui.horizontal(|ui| {
                ui.label("Source:");
                for (mode, label) in [
                    (crate::AudioSourceMode::File,     "File"),
                    (crate::AudioSourceMode::Mic,      "Mic"),
                    (crate::AudioSourceMode::Loopback, "Loopback"),
                    (crate::AudioSourceMode::Silent,   "Silent"),
                ] {
                    let selected = params.audio_source_mode == mode;
                    if ui.selectable_label(selected, label).clicked() && !selected {
                        changes.push(ParamChange::SetAudioSourceMode(mode));
                    }
                }
            });

            // MIDI Shake
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.midi_shake_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::MidiShakeEnabled));
                }
                ui.add_enabled_ui(!params.locks.midi_shake_enabled, |ui| {
                    let mut shake = params.midi_shake_enabled;
                    if ui.checkbox(&mut shake, "MIDI Shake  (Space)").changed() {
                        changes.push(ParamChange::MidiShakeEnabled(shake));
                    }
                });
            });

            // Bass-Zoom Strength
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.bass_zoom_strength).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BassZoomStrength));
                }
                ui.add_enabled_ui(!params.locks.bass_zoom_strength, |ui| {
                    let mut bass = params.bass_zoom_strength;
                    if ui.add(egui::Slider::new(&mut bass, 0.0..=1.0).text("Bass-Zoom Strength").step_by(0.05)).changed() {
                        changes.push(ParamChange::BassZoomStrength(bass));
                    }
                });
            });

            // Beat Reactivity
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.beat_reactivity).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BeatReactivity));
                }
                ui.add_enabled_ui(!params.locks.beat_reactivity, |ui| {
                    let mut react = params.beat_reactivity;
                    if ui.add(egui::Slider::new(&mut react, 0.0..=1.0).text("Beat Reactivity").step_by(0.05)).changed() {
                        changes.push(ParamChange::BeatReactivity(react));
                    }
                });
            });
        });
    }

    fn ribbons_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Ribbons", |ui| {
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.ribbons_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::RibbonsEnabled));
                }
                ui.add_enabled_ui(!params.locks.ribbons_enabled, |ui| {
                    let mut en = params.ribbons_enabled;
                    if ui.checkbox(&mut en, "Enable Ribbons").changed() {
                        changes.push(ParamChange::RibbonsEnabled(en));
                    }
                });
            });

            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.ribbons_intensity).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::RibbonsIntensity));
                }
                ui.add_enabled_ui(!params.locks.ribbons_intensity, |ui| {
                    let mut intensity = params.ribbons_intensity;
                    if ui.add(
                        egui::Slider::new(&mut intensity, 0.0..=2.0)
                            .text("Intensity")
                            .step_by(0.05),
                    ).changed() {
                        changes.push(ParamChange::RibbonsIntensity(intensity));
                    }
                });
            });
        });
    }

    fn palette_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Palette", |ui| {
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.palette_mode).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::PaletteMode));
                }
                ui.add_enabled_ui(!params.locks.palette_mode, |ui| {
                    let mut current = params.palette_mode;
                    egui::ComboBox::from_label("Mode")
                        .selected_text(params.palette_mode.as_str())
                        .show_ui(ui, |ui| {
                            for mode in [
                                crate::PaletteMode::Off,
                                crate::PaletteMode::Warm,
                                crate::PaletteMode::Cool,
                                crate::PaletteMode::Earth,
                                crate::PaletteMode::Neon,
                                crate::PaletteMode::Monochrome,
                            ] {
                                if ui.selectable_value(&mut current, mode, mode.as_str()).clicked() {
                                    changes.push(ParamChange::SetPaletteMode(mode));
                                }
                            }
                        });
                });
            });

            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.palette_tint).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::PaletteTint));
                }
                ui.add_enabled_ui(!params.locks.palette_tint, |ui| {
                    let mut tint = params.palette_tint;
                    if ui.add(egui::Slider::new(&mut tint, 0.0..=1.0).text("Tint").step_by(0.05)).changed() {
                        changes.push(ParamChange::PaletteTint(tint));
                    }
                });
            });

            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.palette_mono_hue).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::PaletteMonoHue));
                }
                ui.add_enabled_ui(
                    !params.locks.palette_mono_hue && params.palette_mode == crate::PaletteMode::Monochrome,
                    |ui| {
                        let mut hue = params.palette_mono_hue;
                        if ui.add(egui::Slider::new(&mut hue, 0.0..=360.0).text("Mono Hue").step_by(1.0)).changed() {
                            changes.push(ParamChange::PaletteMonoHue(hue));
                        }
                    },
                );
            });
        });
    }

    fn blackhole_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Blackhole", |ui| {
            // Enabled
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.blackhole_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BlackholeEnabled));
                }
                ui.add_enabled_ui(!params.locks.blackhole_enabled, |ui| {
                    let mut en = params.blackhole_enabled;
                    if ui.checkbox(&mut en, "Enabled").changed() {
                        changes.push(ParamChange::BlackholeEnabled(en));
                    }
                });
            });

            // blackhole_warp_strength → Feedback Strength
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.blackhole_warp_strength).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BlackholeWarpStrength));
                }
                ui.add_enabled_ui(!params.locks.blackhole_warp_strength, |ui| {
                    let mut v = params.blackhole_warp_strength;
                    if ui.add(egui::Slider::new(&mut v, 0.5..=0.98).text("Feedback Strength").step_by(0.01)).changed() {
                        changes.push(ParamChange::BlackholeWarpStrength(v));
                    }
                });
            });

            // blackhole_warp_curve → Shrink Rate
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.blackhole_warp_curve).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BlackholeWarpCurve));
                }
                ui.add_enabled_ui(!params.locks.blackhole_warp_curve, |ui| {
                    let mut v = params.blackhole_warp_curve;
                    if ui.add(egui::Slider::new(&mut v, 0.85..=0.99).text("Shrink Rate").step_by(0.005)).changed() {
                        changes.push(ParamChange::BlackholeWarpCurve(v));
                    }
                });
            });

            // blackhole_alpha_radius → Edge Alpha
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.blackhole_alpha_radius).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BlackholeAlphaRadius));
                }
                ui.add_enabled_ui(!params.locks.blackhole_alpha_radius, |ui| {
                    let mut v = params.blackhole_alpha_radius;
                    if ui.add(egui::Slider::new(&mut v, 0.0..=1.0).text("Edge Alpha").step_by(0.05)).changed() {
                        changes.push(ParamChange::BlackholeAlphaRadius(v));
                    }
                });
            });

            // blackhole_wander_amount → Wander Amount
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.blackhole_wander_amount).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::BlackholeWanderAmount));
                }
                ui.add_enabled_ui(!params.locks.blackhole_wander_amount, |ui| {
                    let mut v = params.blackhole_wander_amount;
                    if ui.add(egui::Slider::new(&mut v, 0.0..=0.02).text("Wander Amount").step_by(0.001)).changed() {
                        changes.push(ParamChange::BlackholeWanderAmount(v));
                    }
                });
            });
        });
    }

    fn phantom_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Phantom Alpha", |ui| {
            // Enabled (mutually exclusive with blackhole — greyed when blackhole is on)
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.phantom_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::PhantomEnabled));
                }
                ui.add_enabled_ui(!params.locks.phantom_enabled && !params.blackhole_enabled, |ui| {
                    let mut en = params.phantom_enabled;
                    if ui.checkbox(&mut en, "Enabled  (G)").changed() {
                        changes.push(ParamChange::PhantomEnabled(en));
                    }
                });
            });
            if params.blackhole_enabled {
                ui.label(egui::RichText::new("(disabled while Blackhole is on)").small().italics());
            }

            ui.indent("phantom_controls", |ui| {
                // Delay
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_delay_seconds).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomDelaySeconds));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_delay_seconds, |ui| {
                        let mut v = params.phantom_delay_seconds;
                        if ui.add(egui::Slider::new(&mut v, 0.1..=3.0).text("Delay (s)").step_by(0.1)).changed() {
                            changes.push(ParamChange::PhantomDelaySeconds(v));
                        }
                    });
                });

                // Key color
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_key_color).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomKeyColor));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_key_color, |ui| {
                        let mut c = params.phantom_key_color;
                        if egui::color_picker::color_edit_button_rgb(ui, &mut c).changed() {
                            changes.push(ParamChange::PhantomKeyColor(c));
                        }
                        ui.label("Key Color");
                    });
                });

                // Key tolerance
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_key_tolerance).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomKeyTolerance));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_key_tolerance, |ui| {
                        let mut v = params.phantom_key_tolerance;
                        if ui.add(egui::Slider::new(&mut v, 0.0..=0.5).text("Key Tolerance").step_by(0.01)).changed() {
                            changes.push(ParamChange::PhantomKeyTolerance(v));
                        }
                    });
                });

                // Key softness
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_key_softness).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomKeySoftness));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_key_softness, |ui| {
                        let mut v = params.phantom_key_softness;
                        if ui.add(egui::Slider::new(&mut v, 0.0..=0.2).text("Key Softness").step_by(0.005)).changed() {
                            changes.push(ParamChange::PhantomKeySoftness(v));
                        }
                    });
                });

                // Key strength
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_key_strength).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomKeyStrength));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_key_strength, |ui| {
                        let mut v = params.phantom_key_strength;
                        if ui.add(egui::Slider::new(&mut v, 0.0..=1.0).text("Key Strength").step_by(0.05)).changed() {
                            changes.push(ParamChange::PhantomKeyStrength(v));
                        }
                    });
                });

                // Opacity
                ui.horizontal(|ui| {
                    if Self::lock_button(ui, params.locks.phantom_opacity).clicked() {
                        changes.push(ParamChange::ToggleLock(LockTarget::PhantomOpacity));
                    }
                    ui.add_enabled_ui(!params.locks.phantom_opacity, |ui| {
                        let mut v = params.phantom_opacity;
                        if ui.add(egui::Slider::new(&mut v, 0.0..=1.0).text("Opacity").step_by(0.05)).changed() {
                            changes.push(ParamChange::PhantomOpacity(v));
                        }
                    });
                });
            });
        });
    }

    fn color_theory_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Color Theory  (H)", |ui| {

            // Harmony dropdown
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.color_harmony).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorHarmony));
                }
                ui.add_enabled_ui(!params.locks.color_harmony, |ui| {
                    let current = params.color_harmony;
                    egui::ComboBox::from_id_salt("color_harmony_combo")
                        .selected_text(current.name())
                        .show_ui(ui, |ui| {
                            for h in [
                                crate::color::ColorHarmony::Monochromatic,
                                crate::color::ColorHarmony::Analogous,
                                crate::color::ColorHarmony::Complementary,
                                crate::color::ColorHarmony::SplitComplementary,
                                crate::color::ColorHarmony::Triadic,
                                crate::color::ColorHarmony::Tetradic,
                            ] {
                                if ui.selectable_label(current == h, h.name()).clicked() {
                                    changes.push(ParamChange::ColorHarmony(h));
                                }
                            }
                        });
                    ui.label("Harmony");
                });
            });

            // Anchor hue slider + live color swatch
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.color_anchor_hue).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorAnchorHue));
                }
                ui.add_enabled_ui(!params.locks.color_anchor_hue, |ui| {
                    let mut v = params.color_anchor_hue;
                    if ui.add(egui::Slider::new(&mut v, 0.0..=360.0)
                        .text("Anchor hue")
                        .suffix("°")).changed()
                    {
                        changes.push(ParamChange::ColorAnchorHue(v));
                    }
                    let anchor_rgb = crate::color::hsv_to_rgb(
                        params.color_anchor_hue,
                        params.color_saturation,
                        params.color_value,
                    );
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(24.0, 16.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(
                            (anchor_rgb[0] * 255.0) as u8,
                            (anchor_rgb[1] * 255.0) as u8,
                            (anchor_rgb[2] * 255.0) as u8,
                        ),
                    );
                });
            });

            // Saturation slider
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.color_saturation).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorSaturation));
                }
                ui.add_enabled_ui(!params.locks.color_saturation, |ui| {
                    let mut v = params.color_saturation;
                    if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)
                        .text("Saturation").step_by(0.01)).changed()
                    {
                        changes.push(ParamChange::ColorSaturation(v));
                    }
                });
            });

            // Value (brightness) slider
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.color_value).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::ColorValue));
                }
                ui.add_enabled_ui(!params.locks.color_value, |ui| {
                    let mut v = params.color_value;
                    if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)
                        .text("Value").step_by(0.01)).changed()
                    {
                        changes.push(ParamChange::ColorValue(v));
                    }
                });
            });

            // Palette preview: 6 swatches
            ui.horizontal(|ui| {
                ui.label("Preview:");
                let palette = crate::color::palette_from_harmony(
                    params.color_harmony,
                    params.color_anchor_hue,
                    params.color_saturation,
                    params.color_value,
                    6,
                );
                for c in palette {
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(20.0, 20.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(
                            (c[0] * 255.0) as u8,
                            (c[1] * 255.0) as u8,
                            (c[2] * 255.0) as u8,
                        ),
                    );
                }
            });
        });
    }

    fn modes_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Modes", |ui| {
            ui.label("Random Mode  (N)");
            let mut r_en = params.random_mode_enabled;
            if ui.checkbox(&mut r_en, "Enable").changed() {
                changes.push(ParamChange::RandomModeEnabled(r_en));
            }
            ui.add_enabled_ui(r_en, |ui| {
                let mut agg = params.random_mode_aggressiveness;
                if ui
                    .add(egui::Slider::new(&mut agg, 0.0..=1.0).text("Aggressiveness").step_by(0.05))
                    .changed()
                {
                    changes.push(ParamChange::RandomModeAggressiveness(agg));
                }
            });

            ui.separator();

            ui.label("Reactive Mode  (B)");
            let mut re_en = params.reactive_mode_enabled;
            if ui.checkbox(&mut re_en, "Enable").changed() {
                changes.push(ParamChange::ReactiveModeEnabled(re_en));
            }
            ui.add_enabled_ui(re_en, |ui| {
                let mut agg = params.reactive_mode_aggressiveness;
                if ui
                    .add(egui::Slider::new(&mut agg, 0.0..=1.0).text("Aggressiveness").step_by(0.05))
                    .changed()
                {
                    changes.push(ParamChange::ReactiveModeAggressiveness(agg));
                }
            });

            ui.separator();

            ui.label("Party Mode  (Y)");
            let mut p_en = params.party_mode_enabled;
            if ui.checkbox(&mut p_en, "Enable").changed() {
                changes.push(ParamChange::PartyModeEnabled(p_en));
            }
            ui.add_enabled_ui(p_en, |ui| {
                let mut agg = params.party_mode_aggressiveness;
                if ui
                    .add(egui::Slider::new(&mut agg, 0.0..=1.0).text("Aggressiveness").step_by(0.05))
                    .changed()
                {
                    changes.push(ParamChange::PartyModeAggressiveness(agg));
                }
            });

            ui.separator();

            // Beat-Reactive Shake: audio beats drive kick_shake()
            ui.horizontal(|ui| {
                if Self::lock_button(ui, params.locks.audio_shake_enabled).clicked() {
                    changes.push(ParamChange::ToggleLock(LockTarget::AudioShakeEnabled));
                }
                ui.add_enabled_ui(!params.locks.audio_shake_enabled, |ui| {
                    let mut as_en = params.audio_shake_enabled;
                    if ui.checkbox(&mut as_en, "Beat-Reactive Shake").changed() {
                        changes.push(ParamChange::AudioShakeEnabled(as_en));
                    }
                });
            });
        });
    }

    fn audio_player_section(
        ui: &mut egui::Ui,
        player_info: Option<&PlayerInfo>,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Audio Player", |ui| {
            match player_info {
                None => {
                    ui.label("No audio loaded.");
                    ui.label("File → Open Audio…");
                }
                Some(info) => {
                    ui.label(&info.filename);

                    ui.horizontal(|ui| {
                        let play_label = if info.is_playing { "⏸ Pause" } else { "▶ Play" };
                        if ui.button(play_label).clicked() {
                            changes.push(ParamChange::PlayerToggle);
                        }
                        if ui.button("⏹ Stop").clicked() {
                            changes.push(ParamChange::PlayerStop);
                        }
                    });

                    let mut pos = info.position_seconds;
                    let max = info.duration_seconds.max(0.001);
                    if ui
                        .add(egui::Slider::new(&mut pos, 0.0..=max).show_value(false))
                        .changed()
                    {
                        changes.push(ParamChange::PlayerSeek(pos));
                    }

                    ui.label(format!("{:.1}s / {:.1}s", info.position_seconds, info.duration_seconds));
                }
            }
        });
    }

    fn export_section(
        ui: &mut egui::Ui,
        params: &crate::VisualParams,
        player_info: Option<&PlayerInfo>,
        export_progress: Option<ExportProgress>,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Export", |ui| {
            // Resolution dropdown
            ui.horizontal(|ui| {
                ui.label("Resolution");
                let all_resolutions = [
                    crate::ResolutionPreset::SD480,
                    crate::ResolutionPreset::HD720,
                    crate::ResolutionPreset::FullHD,
                    crate::ResolutionPreset::UHD4K,
                ];
                egui::ComboBox::from_id_salt("export_resolution")
                    .selected_text(params.export_resolution.name())
                    .show_ui(ui, |ui| {
                        for preset in all_resolutions {
                            let (w, h) = preset.dimensions();
                            let label = format!("{} ({}×{})", preset.name(), w, h);
                            if ui
                                .selectable_value(
                                    &mut { params.export_resolution },
                                    preset,
                                    label,
                                )
                                .clicked()
                            {
                                changes.push(ParamChange::ExportResolution(preset));
                            }
                        }
                    });
            });

            // Framerate dropdown
            ui.horizontal(|ui| {
                ui.label("Framerate");
                let all_framerates = [
                    crate::FramerateChoice::Fps30,
                    crate::FramerateChoice::Fps60,
                ];
                egui::ComboBox::from_id_salt("export_framerate")
                    .selected_text(params.export_framerate.name())
                    .show_ui(ui, |ui| {
                        for choice in all_framerates {
                            if ui
                                .selectable_value(
                                    &mut { params.export_framerate },
                                    choice,
                                    choice.name(),
                                )
                                .clicked()
                            {
                                changes.push(ParamChange::ExportFramerate(choice));
                            }
                        }
                    });
            });

            // Frame count estimate
            if let Some(info) = player_info {
                let fps = params.export_framerate.fps();
                let total_frames = (info.duration_seconds * fps as f32).ceil() as u64;
                ui.label(format!(
                    "~{} frames  ({:.1}s at {} fps)",
                    total_frames, info.duration_seconds, fps
                ));
            } else {
                ui.label("Load audio to see frame estimate.");
            }

            // Live preview toggle
            let mut lp = params.export_live_preview;
            if ui.checkbox(&mut lp, "Live preview")
                .on_hover_text("Show rendered frames in the window during export. Disable for ~15% faster 4K render.")
                .changed()
            {
                changes.push(ParamChange::SetExportLivePreview(lp));
            }

            ui.separator();

            let audio_loaded = player_info.is_some();
            let is_exporting = export_progress.is_some();
            ui.add_enabled_ui(audio_loaded && !is_exporting, |ui| {
                if ui.button("Export...").clicked() {
                    changes.push(ParamChange::TriggerExport);
                }
            });
            if !audio_loaded {
                ui.label("⚠ Load audio first (File → Open Audio...)");
            }
            if let Some(p) = export_progress {
                if p.is_muxing {
                    ui.add(
                        egui::ProgressBar::new(1.0)
                            .text("Muxing (ffmpeg)…")
                            .animate(true),
                    );
                } else {
                    let fraction = if p.total_frames > 0 {
                        p.current_frame as f32 / p.total_frames as f32
                    } else {
                        0.0
                    };
                    let pct = (fraction * 100.0) as u32;
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .text(format!("{}/{} ({}%)", p.current_frame, p.total_frames, pct)),
                    );
                }
            }
        });
    }

    fn skin_section_static(
        ui: &mut egui::Ui,
        thumbnail: Option<&egui::TextureHandle>,
        thumbnail_aspect: f32,
        crop_y_offset: &mut f32,
        changes: &mut Vec<ParamChange>,
    ) {
        ui.collapsing("Skin", |ui| {
            if let Some(thumb) = thumbnail {
                let [tw, th] = thumb.size();
                let display_w = ui.available_width().min(tw as f32);
                let thumb_aspect = if th > 0 { tw as f32 / th as f32 } else { thumbnail_aspect };
                let display_h = if thumb_aspect > 0.0 { display_w / thumb_aspect } else { display_w };

                let (response, painter) = ui.allocate_painter(
                    egui::vec2(display_w, display_h),
                    egui::Sense::drag(),
                );
                let rect = response.rect;

                // Draw the thumbnail
                painter.image(
                    thumb.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // Crop strip: always 1/16 of display width tall (mirrors crop_skin_image logic)
                let strip_h = (display_w / 16.0).max(1.0);
                let max_top = (display_h - strip_h).max(0.0);
                let strip_top = (*crop_y_offset * max_top).clamp(0.0, max_top);

                let strip_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.min.x, rect.min.y + strip_top),
                    egui::vec2(display_w, strip_h),
                );

                // Dim regions outside the strip
                if strip_rect.min.y > rect.min.y {
                    painter.rect_filled(
                        egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, strip_rect.min.y)),
                        egui::Rounding::ZERO,
                        egui::Color32::from_black_alpha(140),
                    );
                }
                if strip_rect.max.y < rect.max.y {
                    painter.rect_filled(
                        egui::Rect::from_min_max(egui::pos2(rect.min.x, strip_rect.max.y), rect.max),
                        egui::Rounding::ZERO,
                        egui::Color32::from_black_alpha(140),
                    );
                }

                // Gold border around the crop strip
                painter.rect_stroke(
                    strip_rect,
                    egui::Rounding::ZERO,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 220, 80)),
                );

                // Drag to reposition the strip
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let local_y = (pos.y - rect.min.y).clamp(0.0, display_h);
                        let new_top = (local_y - strip_h / 2.0).clamp(0.0, max_top);
                        let new_offset = if max_top > 0.0 { new_top / max_top } else { 0.5 };
                        let new_offset = new_offset.clamp(0.0, 1.0);
                        if (new_offset - *crop_y_offset).abs() > 0.005 {
                            *crop_y_offset = new_offset;
                            changes.push(ParamChange::SkinCropOffset(new_offset));
                        }
                    }
                }

                // Slider as keyboard-accessible alternative
                let mut offset = *crop_y_offset;
                if ui
                    .add(
                        egui::Slider::new(&mut offset, 0.0..=1.0)
                            .text("Vertical offset")
                            .step_by(0.01),
                    )
                    .changed()
                {
                    *crop_y_offset = offset;
                    changes.push(ParamChange::SkinCropOffset(offset));
                }
            } else {
                ui.label("No skin loaded.");
                ui.label("Use File \u{2192} Open Skin\u{2026} to load.");
            }
        });
    }
}
