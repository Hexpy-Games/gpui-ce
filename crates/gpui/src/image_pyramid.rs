use crate::{DevicePixels, ImageId, RenderImage, Size};
use collections::FxHashSet;
use smallvec::SmallVec;
use std::{cell::RefCell, rc::Rc};

pub(crate) fn level_sizes(mut src: Size<DevicePixels>) -> SmallVec<[Size<DevicePixels>; 12]> {
    let mut sizes = SmallVec::new();
    loop {
        src = src.map(|axis| DevicePixels((axis.0 / 2 + axis.0 % 2).max(1)));
        sizes.push(src);
        if src.width.0.max(src.height.0) <= 16 {
            return sizes;
        }
    }
}

pub(crate) fn select_level(
    src: Size<DevicePixels>,
    dst_device: Size<f32>,
    max_level: usize,
) -> usize {
    let ratio =
        (src.width.0 as f32 / dst_device.width).min(src.height.0 as f32 / dst_device.height);
    if ratio < 2.0 {
        0
    } else {
        (ratio.log2().floor() as usize).min(max_level)
    }
}

pub(crate) fn halve_bgra(src: &[u8], size: Size<DevicePixels>) -> (Vec<u8>, Size<DevicePixels>) {
    let dst_size = size.map(|axis| DevicePixels((axis.0 / 2 + axis.0 % 2).max(1)));
    let src_width = usize::from(size.width);
    let src_height = usize::from(size.height);
    let dst_width = usize::from(dst_size.width);
    let dst_height = usize::from(dst_size.height);
    let mut dst = Vec::with_capacity(dst_width * dst_height * 4);

    for y in 0..dst_height {
        for x in 0..dst_width {
            let mut alpha = 0;
            let mut premultiplied = [0; 3];
            for row in [2 * y, (2 * y + 1).min(src_height - 1)] {
                for col in [2 * x, (2 * x + 1).min(src_width - 1)] {
                    let offset = (row * src_width + col) * 4;
                    let pixel_alpha = u32::from(src[offset + 3]);
                    alpha += pixel_alpha;
                    for (channel, sum) in premultiplied.iter_mut().enumerate() {
                        *sum += u32::from(src[offset + channel]) * pixel_alpha;
                    }
                }
            }

            let averaged_alpha = (alpha + 2) / 4;
            if averaged_alpha == 0 {
                dst.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                dst.extend(premultiplied.map(|sum| ((sum + alpha / 2) / alpha) as u8));
                dst.push(averaged_alpha as u8);
            }
        }
    }

    (dst, dst_size)
}

pub(crate) fn build_levels(
    image: &RenderImage,
    frame: usize,
) -> Vec<(Size<DevicePixels>, Vec<u8>)> {
    let mut current_size = image.size(frame);
    let mut pixels = image
        .as_bytes(frame)
        .expect("It's the caller's job to pass a valid frame index");
    let level_count = level_sizes(current_size).len();
    let mut levels = Vec::with_capacity(level_count);
    for _ in 0..level_count {
        let (bytes, next_size) = halve_bgra(pixels, current_size);
        let level = levels.push_mut((next_size, bytes));
        pixels = &level.1;
        current_size = next_size;
    }
    levels
}

pub(crate) type ReadyPyramid = (ImageId, usize, Vec<(Size<DevicePixels>, Vec<u8>)>);

#[derive(Default)]
pub(crate) struct PyramidJobs {
    pending: FxHashSet<(ImageId, usize)>,
    pub(crate) ready: Rc<RefCell<Vec<ReadyPyramid>>>,
}

impl PyramidJobs {
    pub(crate) fn begin(&mut self, key: (ImageId, usize)) -> bool {
        self.pending.insert(key)
    }

    pub(crate) fn cancel(&mut self, image_id: ImageId) {
        self.pending.retain(|(id, _)| *id != image_id);
    }

    pub(crate) fn take_ready(&mut self) -> Vec<ReadyPyramid> {
        let mut ready = self.ready.take();
        ready.retain(|(id, frame, _)| self.pending.remove(&(*id, *frame)));
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size;

    #[test]
    fn select_level_by_shrink_ratio() {
        let src = size(DevicePixels(1446), DevicePixels(1446));
        assert_eq!(select_level(src, size(50.0, 50.0), 7), 4);
        assert_eq!(select_level(src, size(723.0, 723.0), 7), 1);

        let src = size(DevicePixels(199), DevicePixels(199));
        assert_eq!(select_level(src, size(100.0, 100.0), 7), 0);
        assert_eq!(select_level(src, size(200.0, 200.0), 7), 0);
        assert_eq!(select_level(src, size(10.0, 10.0), 2), 2);
        assert_eq!(select_level(src, size(10.0, 50.0), 7), 1);
    }

    #[test]
    fn level_sizes_round_up_and_stop() {
        let sizes = level_sizes(size(DevicePixels(1446), DevicePixels(1000)));
        assert_eq!(
            sizes.as_slice(),
            [
                (723, 500),
                (362, 250),
                (181, 125),
                (91, 63),
                (46, 32),
                (23, 16),
                (12, 8)
            ]
            .map(|(width, height)| size(DevicePixels(width), DevicePixels(height)))
        );
        assert_eq!(
            level_sizes(size(DevicePixels(3), DevicePixels(1))).as_slice(),
            [size(DevicePixels(2), DevicePixels(1))]
        );
    }

    #[test]
    fn halve_bgra_preserves_opaque_pixel_colour() {
        let (pixels, size) = halve_bgra(
            &[31, 91, 201, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            size(DevicePixels(2), DevicePixels(2)),
        );
        assert_eq!(size, Size::new(DevicePixels(1), DevicePixels(1)));
        assert_eq!(pixels, [31, 91, 201, 64]);
    }

    #[test]
    fn cancelled_pyramid_result_is_discarded() {
        let mut jobs = PyramidJobs::default();
        let id = ImageId(0);
        assert!(jobs.begin((id, 0)));
        assert!(!jobs.begin((id, 0)));
        jobs.cancel(id);
        jobs.ready.borrow_mut().push((id, 0, Vec::new()));
        assert!(jobs.take_ready().is_empty());
    }
}
