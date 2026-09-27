//! `wasi:graphics-context`, which `wasi:webgpu@0.0.1` names for canvases and
//! surfaces. This host draws nothing on screen, so the interface is linked
//! only because webgpu's types refer to it, and every function refuses.

use wasmtime::component::Resource;

use crate::{wasi::graphics_context::graphics_context, WasiWebGpuCtx};

/// Never constructed: the constructor refuses.
pub struct GraphicsContext;

/// Never constructed: nothing hands one out.
pub struct AbstractBuffer;

const NO_SURFACES: &str = "wasi:graphics-context is not available: this host runs compute only";

impl graphics_context::Host for WasiWebGpuCtx<'_> {}

impl graphics_context::HostContext for WasiWebGpuCtx<'_> {
    fn new(&mut self) -> wasmtime::Result<Resource<GraphicsContext>> {
        wasmtime::bail!(NO_SURFACES)
    }

    fn get_current_buffer(
        &mut self,
        _context: Resource<GraphicsContext>,
    ) -> wasmtime::Result<Resource<AbstractBuffer>> {
        wasmtime::bail!(NO_SURFACES)
    }

    fn present(&mut self, _context: Resource<GraphicsContext>) -> wasmtime::Result<()> {
        wasmtime::bail!(NO_SURFACES)
    }

    fn drop(&mut self, context: Resource<GraphicsContext>) -> wasmtime::Result<()> {
        self.table.delete(context)?;
        Ok(())
    }
}

impl graphics_context::HostAbstractBuffer for WasiWebGpuCtx<'_> {
    fn drop(&mut self, buffer: Resource<AbstractBuffer>) -> wasmtime::Result<()> {
        self.table.delete(buffer)?;
        Ok(())
    }
}
