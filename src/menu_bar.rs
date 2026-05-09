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
}

pub struct MenuBar {
    pub ctx: Context,
    state: State,
    renderer: Renderer,
    pub pending_actions: Vec<MenuAction>,
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
        Self { ctx, state, renderer, pending_actions: Vec::new() }
    }

    /// Returns true if egui currently wants keyboard input (e.g., a text field is focused).
    #[allow(dead_code)]
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    /// Returns true if egui currently wants pointer input (mouse over a menu, etc).
    #[allow(dead_code)]
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    /// Forward a window event to egui and return whether the visualizer should ignore it.
    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        // Always let egui see the event so menus and hover states update.
        let response = self.state.on_window_event(window, event);

        // Only block the visualizer from events egui actually needs.
        match event {
            winit::event::WindowEvent::KeyboardInput { .. } => {
                // Only consume keyboard when a text field is focused.
                self.ctx.wants_keyboard_input()
            }
            winit::event::WindowEvent::MouseInput { .. }
            | winit::event::WindowEvent::CursorMoved { .. }
            | winit::event::WindowEvent::MouseWheel { .. } => {
                // Consume mouse when the pointer is over a menu or dropdown.
                response.consumed && self.ctx.wants_pointer_input()
            }
            _ => response.consumed,
        }
    }

    /// Drain and return all pending actions accumulated since last call.
    pub fn take_actions(&mut self) -> Vec<MenuAction> {
        std::mem::take(&mut self.pending_actions)
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
    ) {
        let raw_input = self.state.take_egui_input(window);

        let mut frame_actions: Vec<MenuAction> = Vec::new();
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
                        ui.label("(Coming in 24c)");
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
                        ui.label("(Panel toggles coming in 24c)");
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
        });
        self.pending_actions.append(&mut frame_actions);

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
}
