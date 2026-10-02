//! What the stand-in node modules share: the grey mark they track and the
//! few ways they draw on a picture. They are the modules cookbook recipes
//! 145 to 154 name, and the ffrwd-node SDK's examples.

pub mod font;
mod mark;
mod picture;

pub use ffrwd_frame::{Rect, Rgba};
pub use mark::{find, Spot, Spotter};
pub use picture::{text_size, Picture};

/// Six colours a drawing cycles through by id.
pub const PALETTE: [[u8; 4]; 6] = [
    [255, 64, 64, 255],
    [64, 220, 64, 255],
    [64, 128, 255, 255],
    [255, 200, 0, 255],
    [220, 64, 220, 255],
    [0, 220, 220, 255],
];
