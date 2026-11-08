// Modified by HexpyGames for Butler (2026); see NOTICE for the changes.
use gpui::{
    Bounds, CaretMovement, CaretPosition, CaretSelection, CaretSelectionMovement, Pixels, Point,
    ShapedText, SharedString, Size, TextMovement, TextSelectionKind,
};
use std::{ops::Range, sync::Arc};
use unicode_segmentation::UnicodeSegmentation;

/// Data used across successive layout requests to gauge whether layout must be recomputed.
#[derive(Default, Clone)]
pub(super) struct EditableTextLayoutState {
    /// The last known size of the text, as generated during layout.
    pub size: Option<Size<Pixels>>,
    /// The last seen version of `storage` (for tracking when lines need to be reprocessed during layout)
    pub last_seen_storage_version: u16,
    /// Direction and alignment inputs used for this layout.
    pub options: gpui::TextLayoutOptions,
    /// IME composition range underlined in this layout, in laid-out text offsets.
    pub marked_range: Option<Range<usize>>,
    /// The mask character this layout draws in place of each grapheme.
    pub mask: Option<char>,
}

/// Internal state/result after the element has recomputed layout.
#[derive(Default)]
pub(super) struct EditableTextLayoutResult {
    /// Whether the element supports multiple lines of text
    pub supports_multiline: bool,
    /// Whether the element is currently accepting inputs
    pub accepts_input: bool,
    /// The last seen scroll position and size of the element
    pub scroll_bounds: Bounds<Pixels>,
    pub state: EditableTextLayoutState,
    /// The document layout produced by the painter's `prepaint`.
    /// Cached so IME `bounds_for_range` / `character_index_for_point` can evaluate without re-shaping.
    pub document: Option<Arc<ShapedText>>,
    /// Maps storage offsets to offsets in `document`'s text.
    pub display_map: DisplayMap,
    pub line_height: Pixels,
    /// Translation that keeps aligned overflow in the scrollable coordinate space.
    pub document_offset: Point<Pixels>,
    /// The next position the scroll view should move to.
    /// Set by the state in response to user actions.
    pub next_scroll_offset: Option<Point<Pixels>>,
}

/// Maps UTF-8 byte offsets in storage to offsets in the laid-out text, and back.
///
/// Unmasked text is laid out as stored, so both offsets are equal. Masked text is laid out as
/// one mask character per grapheme. Hard line breaks stay line breaks, so lines match storage.
#[derive(Clone, Default)]
pub(super) struct DisplayMap {
    masked: Option<MaskedText>,
}

#[derive(Clone)]
struct MaskedText {
    text: SharedString,
    /// `(storage, display)` offsets of every grapheme start, followed by both text lengths.
    boundaries: Vec<(usize, usize)>,
}

impl DisplayMap {
    /// Maps `text` drawn with `mask` in place of each grapheme, or drawn as stored without a mask.
    pub fn new(text: &str, mask: Option<char>) -> Self {
        let masked = mask.map(|mask| {
            let mut display = String::new();
            let mut boundaries = Vec::new();
            for (offset, grapheme) in text.grapheme_indices(true) {
                boundaries.push((offset, display.len()));
                display.push(if grapheme == "\n" { '\n' } else { mask });
            }
            boundaries.push((text.len(), display.len()));

            MaskedText {
                text: display.into(),
                boundaries,
            }
        });

        Self { masked }
    }

    /// The laid-out text when it is masked, or `None` when it is laid out as stored.
    pub fn masked_text(&self) -> Option<&str> {
        self.masked.as_ref().map(|masked| masked.text.as_str())
    }

    pub fn display_offset(&self, offset: usize) -> usize {
        self.map(offset, |(storage, _)| storage, |(_, display)| display)
    }

    pub fn storage_offset(&self, offset: usize) -> usize {
        self.map(offset, |(_, display)| display, |(storage, _)| storage)
    }

    /// Offsets inside a grapheme snap to its start, and offsets past the end snap to the end.
    fn map(
        &self,
        offset: usize,
        from: fn((usize, usize)) -> usize,
        to: fn((usize, usize)) -> usize,
    ) -> usize {
        let Some(masked) = &self.masked else {
            return offset;
        };
        let index = masked
            .boundaries
            .partition_point(|&boundary| from(boundary) <= offset)
            .saturating_sub(1);

        to(masked.boundaries[index])
    }

    pub fn display_caret(&self, caret: CaretPosition) -> CaretPosition {
        CaretPosition {
            index: self.display_offset(caret.index),
            ..caret
        }
    }

    pub fn storage_caret(&self, caret: CaretPosition) -> CaretPosition {
        CaretPosition {
            index: self.storage_offset(caret.index),
            ..caret
        }
    }

    pub fn display_selection(&self, selection: CaretSelection) -> CaretSelection {
        CaretSelection {
            anchor: self.display_caret(selection.anchor),
            caret: self.display_caret(selection.caret),
        }
    }

    pub fn storage_selection(&self, selection: CaretSelection) -> CaretSelection {
        CaretSelection {
            anchor: self.storage_caret(selection.anchor),
            caret: self.storage_caret(selection.caret),
        }
    }

    pub fn display_range(&self, range: Range<usize>) -> Range<usize> {
        self.display_offset(range.start)..self.display_offset(range.end)
    }

    pub fn storage_range(&self, range: Range<usize>) -> Range<usize> {
        self.storage_offset(range.start)..self.storage_offset(range.end)
    }
}

/// A laid-out document that takes and returns storage offsets.
///
/// Positions cross the [`DisplayMap`] on the way in and out, so a masked layout answers in the
/// same offsets as an unmasked one.
#[derive(Clone, Copy)]
pub(super) struct StorageDocument<'a> {
    pub shaped: &'a ShapedText,
    pub map: &'a DisplayMap,
}

impl StorageDocument<'_> {
    pub fn visual_position_for_caret(
        &self,
        caret: CaretPosition,
        line_height: Pixels,
    ) -> Option<Point<Pixels>> {
        self.shaped
            .visual_position_for_caret(self.map.display_caret(caret), line_height)
    }

    pub fn closest_caret_for_pixel_point(
        &self,
        point: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<CaretPosition, CaretPosition> {
        self.shaped
            .closest_caret_for_pixel_point(point, line_height)
            .map(|caret| self.map.storage_caret(caret))
            .map_err(|caret| self.map.storage_caret(caret))
    }

    pub fn logical_cluster_before(&self, caret: CaretPosition) -> Option<Range<usize>> {
        self.shaped
            .logical_cluster_before(self.map.display_caret(caret))
            .map(|range| self.map.storage_range(range))
    }

    pub fn logical_cluster_after(&self, caret: CaretPosition) -> Option<Range<usize>> {
        self.shaped
            .logical_cluster_after(self.map.display_caret(caret))
            .map(|range| self.map.storage_range(range))
    }

    pub fn caret_movement(
        &self,
        caret: CaretPosition,
        movement: TextMovement,
        vertical_navigation_x: Option<Pixels>,
    ) -> CaretMovement {
        let moved = self.shaped.caret_movement(
            self.map.display_caret(caret),
            movement,
            vertical_navigation_x,
        );

        CaretMovement {
            result: self.map.storage_caret(moved.result),
            ..moved
        }
    }

    pub fn selection_movement(
        &self,
        selection: CaretSelection,
        movement: TextMovement,
        extend: bool,
        vertical_navigation_x: Option<Pixels>,
        line_height: Pixels,
    ) -> CaretSelectionMovement {
        let moved = self.shaped.selection_movement(
            self.map.display_selection(selection),
            movement,
            extend,
            vertical_navigation_x,
            line_height,
        );

        CaretSelectionMovement {
            result: self.map.storage_selection(moved.result),
            ..moved
        }
    }

    pub fn selection_from_pixel_point(
        &self,
        point: Point<Pixels>,
        line_height: Pixels,
        kind: TextSelectionKind,
    ) -> Range<usize> {
        let range = self
            .shaped
            .selection_from_pixel_point(point, line_height, kind);

        self.map.storage_range(range)
    }
}
