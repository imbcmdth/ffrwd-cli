//! One run of the host's own commands on the device: the encoder they are
//! recorded into, the command buffer it finishes as, the submit, and both
//! registry entries let go of on the way out, whatever happened in between.
//!
//! wgpu-core registers the encoder and the finished command buffer as two
//! entries, and keeps each until its owner drops it: the encoder once
//! `finish` has taken its recording, the command buffer once the queue has
//! taken it. wgpu's own wrappers do both in their `Drop`. A host that
//! finishes, submits and walks away leaves two entries behind per frame,
//! about 2 KB each, and this host did: several a frame on a broadcast,
//! 2.5 GB an hour, until the box ran out of memory (2026-10-23).

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{CommandBufferId, CommandEncoderId, QueueId};
use wgpu_core::SubmissionIndex;

/// Commands of the host's own, from an open encoder to a submitted command
/// buffer. Dropping it, finished or not, submitted or not, lets whatever
/// entry it still holds go.
pub(super) struct Commands<'a> {
    global: &'a Global,
    /// What the commands are for, as an error names it: "a readback".
    what: &'a str,
    encoder: Option<CommandEncoderId>,
    buffer: Option<CommandBufferId>,
}

impl<'a> Commands<'a> {
    /// An encoder of the device, open for `what`.
    pub(super) fn open(global: &'a Global, device: &SharedDevice, what: &'a str) -> Result<Self> {
        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        // Made before the check: an encoder that failed is registered too,
        // and goes with the drop.
        let commands = Commands {
            global,
            what,
            encoder: Some(encoder),
            buffer: None,
        };
        if let Some(e) = err {
            bail!("a command encoder for {what}: {e}");
        }
        Ok(commands)
    }

    /// The encoder, while it records.
    pub(super) fn encoder(&self) -> CommandEncoderId {
        self.encoder
            .expect("an encoder is open until it is finished")
    }

    /// Finishes the recording. The command buffer stays this value's own.
    pub(super) fn finish(&mut self) -> Result<CommandBufferId> {
        let Some(encoder) = self.encoder.take() else {
            bail!("{} was finished twice", self.what);
        };
        let (buffer, err) = self.global.command_encoder_finish(
            encoder,
            &wgpu_types::CommandBufferDescriptor { label: None },
            None,
        );
        // The encoder and the command buffer are two entries: the recording
        // moved into the second, the first is done with.
        self.global.command_encoder_drop(encoder);
        self.buffer = Some(buffer);
        if let Some((_, e)) = err {
            bail!("finishing {}: {e}", self.what);
        }
        Ok(buffer)
    }

    /// Finishes the recording where that is still to do, and submits it on
    /// `queue` on its own.
    pub(super) fn submit(self, queue: QueueId) -> Result<SubmissionIndex> {
        Self::submit_all(queue, vec![self])
    }

    /// Submits `all` on `queue` as one submission, in order, finishing any
    /// still recording. Every one is let go of afterwards, submitted or not.
    pub(super) fn submit_all(
        queue: QueueId,
        mut all: Vec<Commands<'a>>,
    ) -> Result<SubmissionIndex> {
        let Some(first) = all.first() else {
            bail!("nothing to submit");
        };
        let (global, what) = (first.global, first.what);
        let mut buffers = Vec::with_capacity(all.len());
        for commands in &mut all {
            buffers.push(match commands.buffer {
                Some(buffer) => buffer,
                None => commands.finish()?,
            });
        }
        global
            .queue_submit(queue, &buffers)
            .map_err(|(_, e)| anyhow!("submitting {what}: {e}"))
        // `all` drops here: a submitted command buffer stays registered
        // until its owner lets it go.
    }
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.global.command_buffer_drop(buffer);
        } else if let Some(encoder) = self.encoder.take() {
            self.global.command_encoder_drop(encoder);
        }
    }
}
