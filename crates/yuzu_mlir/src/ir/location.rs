//! Locations made from a file name that is already an attribute.

use melior::ir::attribute::StringAttribute;
use melior::ir::{AttributeLike, Location};

pub trait LocationExt<'c> {
    /// A range in a file. The name is an attribute made once for the file,
    /// so making a location for each op does not hash the name each time.
    fn file_range(
        file: StringAttribute<'c>,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> Location<'c>;
}

impl<'c> LocationExt<'c> for Location<'c> {
    fn file_range(
        file: StringAttribute<'c>,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> Location<'c> {
        // SAFETY: `file` is a live string attribute of the context the location is made in, and the location is uniqued in that context.
        unsafe {
            Location::from_raw(yuzu_mlir_sys::yzuFileLineColRangeGet(
                file.to_raw(),
                start_line as u32,
                start_column as u32,
                end_line as u32,
                end_column as u32,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::Location;
    use melior::ir::attribute::StringAttribute;

    use super::LocationExt;

    #[test]
    fn a_range_is_the_same_location_the_string_form_makes() {
        let context = crate::context();
        let file = StringAttribute::new(&context, "test.yz");
        assert_eq!(
            Location::file_range(file, 1, 2, 3, 4),
            Location::file_line_col_range(&context, "test.yz", 1, 2, 3, 4)
        );
    }
}
