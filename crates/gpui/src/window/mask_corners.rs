// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
//! The rounded corners of a [`ContentMask`](crate::ContentMask).

use crate::{Corners, Edges};
use std::fmt;

/// The rounded rectangle a [`ContentMask`](crate::ContentMask) clips to, stored relative to the
/// mask's `bounds` and in the same units: the corner radii, and how far the rounded rectangle
/// extends past `bounds` on each side (`bounds` is the intersection of every ancestor's clip,
/// the rounded rectangle is the innermost ancestor that has radii).
///
/// The eight values travel to the shaders as signed 16-bit fixed point (1/16 unit steps, up to
/// ±2048), two per word, adding 16 bytes per masked primitive. Outsets are clamped to the
/// largest radius (a corner whose rectangle corner lies a radius or more past `bounds` never
/// reaches it), so only radii could exceed the range, and only for boxes over 4096 units.
/// The default has no radii and clips to `bounds` alone.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct MaskCorners {
    radii_top: u32,
    radii_bottom: u32,
    outsets_top_right: u32,
    outsets_bottom_left: u32,
}

impl MaskCorners {
    /// Corners with the given radii on a rounded rectangle that extends `outsets` past the
    /// mask's bounds. Without a positive radius this is the plain (default) mask.
    pub fn new(radii: Corners<f32>, outsets: Edges<f32>) -> Self {
        let radii = radii.map(|radius| if *radius > 0.0 { *radius } else { 0.0 });
        let largest = radii
            .top_left
            .max(radii.top_right)
            .max(radii.bottom_right)
            .max(radii.bottom_left);
        if largest <= 0.0 {
            return Self::default();
        }
        let outsets = outsets.map(|outset| outset.min(largest));
        Self {
            radii_top: pack(radii.top_left, radii.top_right),
            radii_bottom: pack(radii.bottom_right, radii.bottom_left),
            outsets_top_right: pack(outsets.top, outsets.right),
            outsets_bottom_left: pack(outsets.bottom, outsets.left),
        }
    }

    /// The four packed words in shader order: top radii, bottom radii, top/right outsets,
    /// bottom/left outsets.
    pub fn packed(&self) -> [u32; 4] {
        [
            self.radii_top,
            self.radii_bottom,
            self.outsets_top_right,
            self.outsets_bottom_left,
        ]
    }

    /// Whether any corner is rounded.
    pub fn is_rounded(&self) -> bool {
        self.radii_top != 0 || self.radii_bottom != 0
    }

    /// The corner radii.
    pub fn radii(&self) -> Corners<f32> {
        let (top_left, top_right) = unpack(self.radii_top);
        let (bottom_right, bottom_left) = unpack(self.radii_bottom);
        Corners {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        }
    }

    /// How far the rounded rectangle extends past the mask's bounds on each side.
    pub fn outsets(&self) -> Edges<f32> {
        let (top, right) = unpack(self.outsets_top_right);
        let (bottom, left) = unpack(self.outsets_bottom_left);
        Edges {
            top,
            right,
            bottom,
            left,
        }
    }

    /// The same rounded rectangle, expressed against `to` instead of `from` (both given as
    /// rectangle edges: top, right, bottom, left).
    pub fn rebased(&self, from: Edges<f32>, to: Edges<f32>) -> Self {
        if !self.is_rounded() {
            return *self;
        }
        let outsets = self.outsets();
        Self::new(
            self.radii(),
            Edges {
                top: outsets.top + to.top - from.top,
                right: outsets.right + from.right - to.right,
                bottom: outsets.bottom + from.bottom - to.bottom,
                left: outsets.left + to.left - from.left,
            },
        )
    }

    /// The corners in units `factor` times larger.
    pub fn scale(&self, factor: f32) -> Self {
        if !self.is_rounded() {
            return *self;
        }
        Self::new(
            self.radii().map(|radius| radius * factor),
            self.outsets().map(|outset| outset * factor),
        )
    }

    /// Whether the rounded corners clip anything inside `visible`, for a mask whose bounds have
    /// the given edges (all as top, right, bottom, left).
    pub fn clips(&self, bounds: Edges<f32>, visible: Edges<f32>) -> bool {
        if !self.is_rounded() {
            return false;
        }
        let radii = self.radii();
        let outsets = self.outsets();
        let (top, right) = (bounds.top - outsets.top, bounds.right + outsets.right);
        let (bottom, left) = (bounds.bottom + outsets.bottom, bounds.left - outsets.left);
        let reaches =
            |radius: f32, horizontal: bool, vertical: bool| radius > 0.0 && horizontal && vertical;
        reaches(
            radii.top_left,
            visible.left < left + radii.top_left,
            visible.top < top + radii.top_left,
        ) || reaches(
            radii.top_right,
            visible.right > right - radii.top_right,
            visible.top < top + radii.top_right,
        ) || reaches(
            radii.bottom_right,
            visible.right > right - radii.bottom_right,
            visible.bottom > bottom - radii.bottom_right,
        ) || reaches(
            radii.bottom_left,
            visible.left < left + radii.bottom_left,
            visible.bottom > bottom - radii.bottom_left,
        )
    }
}

impl fmt::Debug for MaskCorners {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaskCorners")
            .field("radii", &self.radii())
            .field("outsets", &self.outsets())
            .finish()
    }
}

/// Steps per unit of the fixed-point values.
const STEPS: f32 = 16.0;

/// Two values as signed 16-bit fixed point, `low` in the low half.
fn pack(low: f32, high: f32) -> u32 {
    u32::from(fixed(low)) | (u32::from(fixed(high)) << 16)
}

fn unpack(packed: u32) -> (f32, f32) {
    let value = |half: u32| f32::from(half as u16 as i16) / STEPS;
    (value(packed), value(packed >> 16))
}

/// Rounds to the nearest step, saturating at the 16-bit range.
fn fixed(value: f32) -> u16 {
    (value * STEPS)
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16 as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(top: f32, right: f32, bottom: f32, left: f32) -> Edges<f32> {
        Edges {
            top,
            right,
            bottom,
            left,
        }
    }

    #[test]
    fn fixed_point_round_trips_to_a_sixteenth_and_saturates() {
        for value in [0.0, 0.5, 12.0, 16.25, 300.0625, -0.75, -2.0] {
            assert_eq!(unpack(pack(value, -value)), (value, -value));
        }
        assert_eq!(unpack(pack(16.01, 5000.0)), (16.0, 2047.9375));
    }

    #[test]
    fn rebasing_onto_a_smaller_rect_keeps_the_rounded_rect_in_place() {
        let corners = MaskCorners::new(Corners::all(16.0), Edges::default());
        let rebased = corners.rebased(edges(0.0, 100.0, 80.0, 0.0), edges(4.0, 100.0, 70.0, 2.0));
        assert_eq!(rebased.radii(), Corners::all(16.0));
        assert_eq!(rebased.outsets(), edges(4.0, 0.0, 10.0, 2.0));
    }

    #[test]
    fn outsets_beyond_every_radius_are_clamped_and_unrounded_masks_stay_default() {
        let corners = MaskCorners::new(Corners::all(8.0), edges(500.0, 0.0, 0.0, 0.0));
        assert_eq!(corners.outsets().top, 8.0);
        assert_eq!(
            MaskCorners::new(Corners::all(-1.0), edges(3.0, 0.0, 0.0, 0.0)),
            MaskCorners::default()
        );
    }

    #[test]
    fn clips_only_where_a_corner_arc_reaches_the_visible_rect() {
        let corners = MaskCorners::new(Corners::all(16.0), Edges::default());
        let bounds = edges(0.0, 100.0, 100.0, 0.0);
        assert!(corners.clips(bounds, edges(0.0, 50.0, 50.0, 0.0)));
        assert!(!corners.clips(bounds, edges(16.0, 84.0, 84.0, 16.0)));
        assert!(!corners.clips(bounds, edges(0.0, 84.0, 100.0, 16.0)));
    }
}
