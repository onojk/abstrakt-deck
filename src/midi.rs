use crossbeam_channel::{bounded, Receiver};
use midir::{Ignore, MidiInput, MidiInputConnection};

#[derive(Debug, Clone, Copy)]
pub enum MidiEvent {
    ControlChange(u8, u8),
    NoteOn(u8, u8),
    NoteOff(#[allow(dead_code)] u8),
}

pub struct MidiCapture {
    pub event_rx: Receiver<MidiEvent>,
    _connection: MidiInputConnection<()>,
}

impl MidiCapture {
    pub fn start() -> Result<Self, String> {
        let mut midi_in = MidiInput::new("abstrakt-deck")
            .map_err(|e| format!("Failed to create MIDI input: {}", e))?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            return Err("No MIDI input ports available".to_string());
        }

        for p in &ports {
            let name = midi_in.port_name(p).unwrap_or_else(|_| "<unknown>".to_string());
            log::info!("MIDI port available: {}", name);
        }

        // Priority: VMPK by name → "MIDI Out" (VMPK's actual port label) → non-Through → first port
        let port = ports
            .iter()
            .find(|p| {
                midi_in
                    .port_name(p)
                    .map(|n| n.to_lowercase().contains("vmpk"))
                    .unwrap_or(false)
            })
            .or_else(|| {
                ports.iter().find(|p| {
                    midi_in
                        .port_name(p)
                        .map(|n| n.to_lowercase().contains("midi out"))
                        .unwrap_or(false)
                })
            })
            .or_else(|| {
                ports.iter().find(|p| {
                    midi_in
                        .port_name(p)
                        .map(|n| !n.to_lowercase().contains("through"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(&ports[0]);

        let port_name = midi_in
            .port_name(port)
            .unwrap_or_else(|_| "<unknown>".to_string());
        log::info!("MIDI port selected: {}", port_name);

        let (event_tx, event_rx) = bounded::<MidiEvent>(256);

        let connection = midi_in
            .connect(
                port,
                "abstrakt-deck-midi",
                move |_timestamp, message, _| {
                    if message.is_empty() { return; }
                    let status = message[0] & 0xF0;

                    let event = match status {
                        0x80 if message.len() >= 2 => Some(MidiEvent::NoteOff(message[1])),
                        0x90 if message.len() >= 3 => {
                            if message[2] == 0 {
                                Some(MidiEvent::NoteOff(message[1]))
                            } else {
                                Some(MidiEvent::NoteOn(message[1], message[2]))
                            }
                        }
                        0xB0 if message.len() >= 3 => {
                            Some(MidiEvent::ControlChange(message[1], message[2]))
                        }
                        _ => None,
                    };

                    if let Some(e) = event {
                        let _ = event_tx.try_send(e);
                    }
                },
                (),
            )
            .map_err(|e| format!("Failed to connect to MIDI port: {}", e))?;

        log::info!("MIDI capture started");
        Ok(Self {
            event_rx,
            _connection: connection,
        })
    }
}
