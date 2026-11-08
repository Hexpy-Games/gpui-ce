//! Looping sprites drawn by Core Animation instead of the GPU frame.
//!
//! The animations read the same clock as the shaders (`frame_tile_bounds` and `turn_about` in
//! gpui_render), so a sprite looks the same on either path.

use core_graphics::{
    base::{
        kCGBitmapByteOrder32Little, kCGImageAlphaPremultipliedFirst, kCGRenderingIntentDefault,
    },
    color_space::CGColorSpace,
    data_provider::CGDataProvider,
    image::CGImage,
};
use foreign_types::ForeignType as _;
use gpui::{
    CompositorSprite, FRAME_GUTTER, FrameLoop, MotionEasing, MotionRepeat, SpriteContents,
    animation_time_ms, hsla_to_rgba,
};
use gpui_wgpu::WgpuAtlas;
use objc2::{
    Encode, Encoding, class, msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool},
};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};
use objc2_quartz_core::{CAMediaTimingFunction, CATransform3D};
use std::{collections::HashMap, f64::consts::TAU, ffi::c_void, sync::Arc};

type ObjcId = *mut AnyObject;

#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {
    fn CACurrentMediaTime() -> f64;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGColorCreate(space: *mut c_void, components: *const f64) -> *mut c_void;
    fn CGColorRelease(color: *mut c_void);
}

/// A `CGColorRef` argument.
#[repr(transparent)]
#[derive(Clone, Copy)]
struct CgColor(*mut c_void);

// SAFETY: `CgColor` is a pointer to the opaque `CGColor` struct.
unsafe impl Encode for CgColor {
    const ENCODING: Encoding = Encoding::Pointer(&Encoding::Struct("CGColor", &[]));
}

/// Core Animation objects are only touched on the main thread, where windows draw.
pub(super) struct MainThread<T>(pub(super) T);

// SAFETY: see above; the renderer is only used from the main thread.
unsafe impl<T> Send for MainThread<T> {}

/// Where the root layer puts a point, as measured when a sprite layer was made.
#[derive(Clone, Copy, PartialEq)]
struct Placement {
    scale: f64,
    /// Height of the root layer, in points.
    height: f64,
    flipped: bool,
}

struct Entry {
    sprite: CompositorSprite,
    placement: Placement,
    /// The clip layer, a sublayer of the root.
    layer: MainThread<Retained<AnyObject>>,
}

/// The sublayers of the window's layer that draw the scene's [`CompositorSprite`]s.
#[derive(Default)]
pub struct CompositorLayers {
    entries: Vec<Entry>,
    /// The cell images of each flipbook grid in use, by (texture, tile id).
    cells: HashMap<(gpui::AtlasTextureId, u32), MainThread<Vec<CGImage>>>,
}

impl CompositorLayers {
    /// Whether `update` with `sprites` would add, remove or change a layer.
    pub fn changed(&self, root: ObjcId, sprites: &[CompositorSprite]) -> bool {
        if sprites.is_empty() && self.entries.is_empty() {
            return false;
        }
        let matched = self.matches(sprites, placement(root));
        matched.len() != self.entries.len() || matched.iter().any(Option::is_none)
    }

    /// Makes the sublayers of `root` draw exactly `sprites`. A layer whose sprite is unchanged is
    /// left alone, so its animation keeps running.
    pub fn update(&mut self, root: ObjcId, sprites: &[CompositorSprite], atlas: &WgpuAtlas) {
        let place = placement(root);
        let matched = self.matches(sprites, place);
        if matched.len() == self.entries.len() && matched.iter().all(Option::is_some) {
            return;
        }

        let now_ms = animation_time_ms();
        let mut old: Vec<Option<Entry>> = std::mem::take(&mut self.entries)
            .into_iter()
            .map(Some)
            .collect();
        let mut entries = Vec::with_capacity(sprites.len());
        let _: () = unsafe { msg_send![class!(CATransaction), begin] };
        let _: () = unsafe { msg_send![class!(CATransaction), setDisableActions: Bool::YES] };
        for (sprite, found) in sprites.iter().zip(matched) {
            if let Some(entry) = found.and_then(|index| old[index].take()) {
                entries.push(entry);
            } else {
                let pixels;
                let cells = match &sprite.contents {
                    SpriteContents::Atlas(_) => {
                        let Some(cells) = self.cells_for(sprite, atlas) else {
                            continue;
                        };
                        cells
                    }
                    SpriteContents::Pixels { size, bytes } => {
                        pixels = vec![CGImage::new(
                            size.width.0 as usize,
                            size.height.0 as usize,
                            8,
                            32,
                            size.width.0 as usize * 4,
                            &CGColorSpace::create_device_rgb(),
                            kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little,
                            &CGDataProvider::from_buffer(Arc::new(bytes.clone())),
                            false,
                            kCGRenderingIntentDefault,
                        )];
                        &pixels
                    }
                };
                let layer = make_layers(sprite, cells, place, now_ms);
                let _: () = unsafe { msg_send![root, addSublayer: &*layer] };
                entries.push(Entry {
                    sprite: sprite.clone(),
                    placement: place,
                    layer: MainThread(layer),
                });
            }
        }
        for entry in old.into_iter().flatten() {
            let _: () = unsafe { msg_send![&*entry.layer.0, removeFromSuperlayer] };
        }
        let _: () = unsafe { msg_send![class!(CATransaction), commit] };

        self.entries = entries;
        let entries = &self.entries;
        self.cells.retain(|key, _| {
            entries
                .iter()
                .any(|entry| matches!(&entry.sprite.contents, SpriteContents::Atlas(tile) if (tile.texture_id, tile.tile_id.0) == *key))
        });
    }

    /// For each of `sprites`, the existing entry that already draws it.
    fn matches(&self, sprites: &[CompositorSprite], place: Placement) -> Vec<Option<usize>> {
        let mut taken = vec![false; self.entries.len()];
        sprites
            .iter()
            .map(|sprite| {
                let index = self.entries.iter().enumerate().position(|(index, entry)| {
                    !taken[index] && entry.sprite == *sprite && entry.placement == place
                })?;
                taken[index] = true;
                Some(index)
            })
            .collect()
    }

    fn cells_for(&mut self, sprite: &CompositorSprite, atlas: &WgpuAtlas) -> Option<&[CGImage]> {
        let SpriteContents::Atlas(tile) = &sprite.contents else {
            return None;
        };
        let key = (tile.texture_id, tile.tile_id.0);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.cells.entry(key) {
            let (size, bytes) = atlas.frame_grid(tile)?;
            let (width, height) = (size.width.0 as usize, size.height.0 as usize);
            let cells = crop_cells(
                (width, height),
                &bytes,
                &sprite.frames,
                sprite.color.is_some(),
            )?;
            entry.insert(MainThread(cells));
        }
        self.cells.get(&key).map(|cells| cells.0.as_slice())
    }
}

fn placement(root: ObjcId) -> Placement {
    let bounds: NSRect = unsafe { msg_send![root, bounds] };
    let scale: f64 = unsafe { msg_send![root, contentsScale] };
    let flipped: Bool = unsafe { msg_send![root, isGeometryFlipped] };
    Placement {
        scale: scale.max(1.0),
        height: bounds.size.height,
        flipped: flipped.as_bool(),
    }
}

/// One image per cell of the `grid` (width, height) of `frames`, without its gutter, as
/// premultiplied BGRA. A monochrome grid becomes a black image carrying the alpha, to mask with.
fn crop_cells(
    grid: (usize, usize),
    bytes: &[u8],
    frames: &FrameLoop,
    monochrome: bool,
) -> Option<Vec<CGImage>> {
    let (grid_width, grid_height) = grid;
    let bytes_per_pixel = if monochrome { 1 } else { 4 };
    let count = frames.frame_count.max(1) as usize;
    let columns = (frames.columns as usize).max(1);
    let rows = count.div_ceil(columns);
    let gutter = FRAME_GUTTER as usize;
    let (cell_width, cell_height) = (grid_width / columns, grid_height / rows);
    if bytes.len() != grid_width * grid_height * bytes_per_pixel
        || cell_width <= 2 * gutter
        || cell_height <= 2 * gutter
    {
        return None;
    }
    let (width, height) = (cell_width - 2 * gutter, cell_height - 2 * gutter);

    Some(
        (0..count)
            .map(|index| {
                let x = (index % columns) * cell_width + gutter;
                let y = (index / columns) * cell_height + gutter;
                let mut pixels = Vec::with_capacity(width * height * 4);
                for row in 0..height {
                    let start = ((y + row) * grid_width + x) * bytes_per_pixel;
                    let line = &bytes[start..start + width * bytes_per_pixel];
                    if monochrome {
                        for &alpha in line {
                            pixels.extend_from_slice(&[0, 0, 0, alpha]);
                        }
                    } else {
                        for pixel in line.chunks_exact(4) {
                            let alpha = u32::from(pixel[3]);
                            pixels.extend(
                                pixel[..3].iter().map(|&channel| {
                                    ((u32::from(channel) * alpha + 127) / 255) as u8
                                }),
                            );
                            pixels.push(pixel[3]);
                        }
                    }
                }
                CGImage::new(
                    width,
                    height,
                    8,
                    32,
                    width * 4,
                    &CGColorSpace::create_device_rgb(),
                    kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little,
                    &CGDataProvider::from_buffer(Arc::new(pixels)),
                    false,
                    kCGRenderingIntentDefault,
                )
            })
            .collect(),
    )
}

fn rect(x: f64, y: f64, width: f64, height: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(width, height))
}

fn new_layer() -> Retained<AnyObject> {
    unsafe { msg_send![class!(CALayer), layer] }
}

/// The clip layer of `sprite`, holding the layer that shows its cells.
///
/// The clip and sprite layers are not flipped, so a positive rotation turns counterclockwise on
/// screen; the turn runs the other way.
fn make_layers(
    sprite: &CompositorSprite,
    cells: &[CGImage],
    place: Placement,
    now_ms: u32,
) -> Retained<AnyObject> {
    let points = |value: f32| f64::from(value) / place.scale;
    let (bounds, clip) = (sprite.bounds, sprite.clip);
    let clip_height = points(clip.size.height.0);
    let clip_top = points(clip.origin.y.0);
    let clip_y = if place.flipped {
        clip_top
    } else {
        place.height - clip_top - clip_height
    };
    let width = points(bounds.size.width.0);
    let height = points(bounds.size.height.0);
    let x = points(bounds.origin.x.0 - clip.origin.x.0);
    let y = clip_height - points(bounds.origin.y.0 - clip.origin.y.0) - height;

    let clip_layer = new_layer();
    let _: () = unsafe { msg_send![&*clip_layer, setMasksToBounds: Bool::YES] };
    let _: () = unsafe {
        msg_send![
            &*clip_layer,
            setFrame: rect(points(clip.origin.x.0), clip_y, points(clip.size.width.0), clip_height)
        ]
    };

    let layer = new_layer();
    let _: () = unsafe { msg_send![&*layer, setContentsScale: place.scale] };
    let _: () = unsafe { msg_send![&*layer, setFrame: rect(x, y, width, height)] };

    // The layer showing the cells: the sprite itself, or the mask of a tinted one.
    let image_layer = match sprite.color {
        Some(color) => {
            let rgba = hsla_to_rgba(color);
            let components = [
                f64::from(rgba.red),
                f64::from(rgba.green),
                f64::from(rgba.blue),
                f64::from(rgba.alpha),
            ];
            let space = CGColorSpace::create_device_rgb();
            let tint = unsafe { CGColorCreate(space.as_ptr().cast(), components.as_ptr()) };
            let _: () = unsafe { msg_send![&*layer, setBackgroundColor: CgColor(tint)] };
            unsafe { CGColorRelease(tint) };

            let mask = new_layer();
            let _: () = unsafe { msg_send![&*mask, setContentsScale: place.scale] };
            let _: () = unsafe { msg_send![&*mask, setFrame: rect(0.0, 0.0, width, height)] };
            let _: () = unsafe { msg_send![&*layer, setMask: &*mask] };
            mask
        }
        None => layer.clone(),
    };
    let first = cells[0].as_ptr() as ObjcId;
    let _: () = unsafe { msg_send![&*image_layer, setContents: first] };

    let frames = sprite.frames;
    if frames.frame_count > 1 {
        let values: Retained<AnyObject> = unsafe { msg_send![class!(NSMutableArray), array] };
        for cell in cells {
            let _: () = unsafe { msg_send![&*values, addObject: cell.as_ptr() as ObjcId] };
        }
        let animation: Retained<AnyObject> = unsafe {
            msg_send![
                class!(CAKeyframeAnimation),
                animationWithKeyPath: &*NSString::from_str("contents")
            ]
        };
        let _: () = unsafe { msg_send![&*animation, setValues: &*values] };
        let _: () =
            unsafe { msg_send![&*animation, setCalculationMode: &*NSString::from_str("discrete")] };
        loop_forever(&animation, frames.period_ms, frames.start_ms, now_ms);
        add_animation(&image_layer, &animation, "flipbook");
    }
    if frames.turn_period_ms != 0 {
        let number = |value: f64| -> Retained<AnyObject> {
            unsafe { msg_send![class!(NSNumber), numberWithDouble: value] }
        };
        let animation: Retained<AnyObject> = unsafe {
            msg_send![
                class!(CABasicAnimation),
                animationWithKeyPath: &*NSString::from_str("transform.rotation.z")
            ]
        };
        let _: () = unsafe { msg_send![&*animation, setFromValue: &*number(0.0)] };
        let _: () = unsafe { msg_send![&*animation, setToValue: &*number(-TAU)] };
        loop_forever(&animation, frames.turn_period_ms, frames.start_ms, now_ms);
        add_animation(&layer, &animation, "turn");
    }

    let motion = sprite.motion;
    if motion.is_animated() {
        let anchor = NSPoint::new(f64::from(motion.origin.x), 1.0 - f64::from(motion.origin.y));
        let _: () = unsafe { msg_send![&*layer, setAnchorPoint: anchor] };
        let _: () = unsafe {
            msg_send![&*layer, setPosition: NSPoint::new(x + width * anchor.x, y + height * anchor.y)]
        };
        let mask = motion.clip.map(|(from, _)| {
            let mask = new_layer();
            let space = CGColorSpace::create_device_rgb();
            let color = unsafe { CGColorCreate(space.as_ptr().cast(), [0.0, 0.0, 0.0, 1.0].as_ptr()) };
            let _: () = unsafe { msg_send![&*mask, setBackgroundColor: CgColor(color)] };
            unsafe { CGColorRelease(color) };
            let _: () = unsafe {
                msg_send![&*mask, setFrame: rect(f64::from(f32::from(from.left)), f64::from(f32::from(from.bottom)), width - f64::from(f32::from(from.left + from.right)), height - f64::from(f32::from(from.top + from.bottom)))]
            };
            let _: () = unsafe { msg_send![&*layer, setMask: &*mask] };
            mask
        });
        for (key, target) in [("opacity", &layer), ("transform", &layer)]
            .into_iter()
            .chain(
                mask.as_ref()
                    .into_iter()
                    .flat_map(|mask| [("bounds", mask), ("position", mask)]),
            )
        {
            let value = |progress: f32| -> Retained<AnyObject> {
                if key == "opacity" {
                    let opacity = f64::from(
                        motion.opacity.0 + (motion.opacity.1 - motion.opacity.0) * progress,
                    );
                    unsafe { msg_send![class!(NSNumber), numberWithDouble: opacity] }
                } else if key == "bounds" || key == "position" {
                    let (from, to) = motion.clip.unwrap();
                    let left = f64::from(f32::from(from.left + (to.left - from.left) * progress));
                    let right =
                        f64::from(f32::from(from.right + (to.right - from.right) * progress));
                    let top = f64::from(f32::from(from.top + (to.top - from.top) * progress));
                    let bottom = f64::from(f32::from(
                        from.bottom + (to.bottom - from.bottom) * progress,
                    ));
                    if key == "bounds" {
                        unsafe {
                            msg_send![class!(NSValue), valueWithRect: rect(0.0, 0.0, width - left - right, height - top - bottom)]
                        }
                    } else {
                        unsafe {
                            msg_send![class!(NSValue), valueWithPoint: NSPoint::new((left + width - right) * 0.5, (bottom + height - top) * 0.5)]
                        }
                    }
                } else {
                    let offset = motion.offset.0 + (motion.offset.1 - motion.offset.0) * progress;
                    let scale =
                        f64::from(motion.scale.0 + (motion.scale.1 - motion.scale.0) * progress);
                    let transform = CATransform3D {
                        m11: scale,
                        m12: 0.0,
                        m13: 0.0,
                        m14: 0.0,
                        m21: 0.0,
                        m22: scale,
                        m23: 0.0,
                        m24: 0.0,
                        m31: 0.0,
                        m32: 0.0,
                        m33: 1.0,
                        m34: 0.0,
                        m41: f64::from(f32::from(offset.x)),
                        m42: -f64::from(f32::from(offset.y)),
                        m43: 0.0,
                        m44: 1.0,
                    };
                    unsafe { msg_send![class!(NSValue), valueWithCATransform3D: transform] }
                }
            };
            let animation: Retained<AnyObject> = match motion.easing {
                MotionEasing::CubicBezier([x1, y1, x2, y2]) => {
                    let animation: Retained<AnyObject> = unsafe {
                        msg_send![class!(CABasicAnimation), animationWithKeyPath: &*NSString::from_str(key)]
                    };
                    let _: () = unsafe { msg_send![&*animation, setFromValue: &*value(0.0)] };
                    let _: () = unsafe { msg_send![&*animation, setToValue: &*value(1.0)] };
                    let timing = CAMediaTimingFunction::functionWithControlPoints(x1, y1, x2, y2);
                    let _: () = unsafe { msg_send![&*animation, setTimingFunction: &*timing] };
                    animation
                }
                MotionEasing::Linear { points, len } => {
                    let animation: Retained<AnyObject> = unsafe {
                        msg_send![class!(CAKeyframeAnimation), animationWithKeyPath: &*NSString::from_str(key)]
                    };
                    let values: Retained<AnyObject> =
                        unsafe { msg_send![class!(NSMutableArray), array] };
                    let times: Retained<AnyObject> =
                        unsafe { msg_send![class!(NSMutableArray), array] };
                    for &(x, y) in &points[..usize::from(len)] {
                        let _: () = unsafe { msg_send![&*values, addObject: &*value(y)] };
                        let time: Retained<AnyObject> =
                            unsafe { msg_send![class!(NSNumber), numberWithDouble: f64::from(x)] };
                        let _: () = unsafe { msg_send![&*times, addObject: &*time] };
                    }
                    let _: () = unsafe { msg_send![&*animation, setValues: &*values] };
                    let _: () = unsafe { msg_send![&*animation, setKeyTimes: &*times] };
                    let _: () = unsafe {
                        msg_send![&*animation, setCalculationMode: &*NSString::from_str("linear")]
                    };
                    animation
                }
                MotionEasing::Steps(n) => {
                    let animation: Retained<AnyObject> = unsafe {
                        msg_send![class!(CAKeyframeAnimation), animationWithKeyPath: &*NSString::from_str(key)]
                    };
                    let values: Retained<AnyObject> =
                        unsafe { msg_send![class!(NSMutableArray), array] };
                    let times: Retained<AnyObject> =
                        unsafe { msg_send![class!(NSMutableArray), array] };
                    for i in 0..=n {
                        let _: () =
                            unsafe { msg_send![&*values, addObject: &*value(i as f32 / n as f32)] };
                        let time: Retained<AnyObject> = unsafe {
                            msg_send![class!(NSNumber), numberWithDouble: f64::from(i) / f64::from(n)]
                        };
                        let _: () = unsafe { msg_send![&*times, addObject: &*time] };
                    }
                    let _: () = unsafe { msg_send![&*animation, setValues: &*values] };
                    let _: () = unsafe { msg_send![&*animation, setKeyTimes: &*times] };
                    let _: () = unsafe {
                        msg_send![&*animation, setCalculationMode: &*NSString::from_str("discrete")]
                    };
                    animation
                }
            };
            let begin = unsafe { CACurrentMediaTime() }
                - f64::from(now_ms.wrapping_sub(motion.start_ms)) / 1000.0;
            let _: () = unsafe { msg_send![&*animation, setBeginTime: begin] };
            let _: () = unsafe {
                msg_send![&*animation, setDuration: f64::from(motion.duration_ms) / 1000.0]
            };
            let _: () = unsafe { msg_send![&*animation, setRepeatCount: f32::INFINITY] };
            let _: () = unsafe {
                msg_send![&*animation, setAutoreverses: Bool::new(motion.repeat == MotionRepeat::Alternate)]
            };
            let _: () = unsafe { msg_send![&*animation, setRemovedOnCompletion: Bool::NO] };
            add_animation(target, &animation, key);
        }
    }

    let _: () = unsafe { msg_send![&*clip_layer, addSublayer: &*layer] };
    clip_layer
}

/// Makes `animation` repeat every `period_ms` and be `now_ms - start_ms` into its loop now.
fn loop_forever(animation: &AnyObject, period_ms: u32, start_ms: u32, now_ms: u32) {
    let period_ms = period_ms.max(1);
    let elapsed_ms = now_ms.wrapping_sub(start_ms) % period_ms;
    let begin = unsafe { CACurrentMediaTime() } - f64::from(elapsed_ms) / 1000.0;
    let _: () = unsafe { msg_send![animation, setBeginTime: begin] };
    let _: () = unsafe { msg_send![animation, setDuration: f64::from(period_ms) / 1000.0] };
    let _: () = unsafe { msg_send![animation, setRepeatCount: f32::INFINITY] };
    let _: () = unsafe { msg_send![animation, setRemovedOnCompletion: Bool::NO] };
}

fn add_animation(layer: &AnyObject, animation: &AnyObject, key: &str) {
    let _: () =
        unsafe { msg_send![layer, addAnimation: animation, forKey: &*NSString::from_str(key)] };
}
