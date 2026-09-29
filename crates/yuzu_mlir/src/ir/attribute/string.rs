//! Names interned in the context as string attributes.

use std::fmt::{self, Write};

use melior::Context;
use melior::ir::attribute::StringAttribute;

/// The longest name formatted on the stack.
const STACK: usize = 128;

/// Interns a formatted name in the context. A name of up to `STACK` bytes
/// is formatted on the stack, so it costs no allocation; a longer one is
/// formatted into a `String`.
#[must_use]
pub fn intern_fmt<'c>(context: &'c Context, args: fmt::Arguments<'_>) -> &'c str {
    let mut buffer = StackBuffer {
        bytes: [0; STACK],
        len: 0,
    };
    if buffer.write_fmt(args).is_ok() {
        return StringAttribute::new(context, buffer.as_str()).value();
    }

    StringAttribute::new(context, &fmt::format(args)).value()
}

struct StackBuffer {
    bytes: [u8; STACK],
    len: usize,
}

impl StackBuffer {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("only whole `str`s are written")
    }
}

impl fmt::Write for StackBuffer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        let slot = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        slot.copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{STACK, intern_fmt};

    #[test]
    fn a_short_and_a_long_name_intern_alike() {
        let context = crate::context();
        let short = intern_fmt(&context, format_args!("{}.{}", "helpers", 2));
        assert_eq!(short, "helpers.2");

        let long = "x".repeat(STACK);
        let interned = intern_fmt(&context, format_args!("{long}.{}", 3));
        assert_eq!(interned, format!("{long}.3"));
        assert_eq!(
            interned.as_ptr(),
            intern_fmt(&context, format_args!("{long}.3")).as_ptr()
        );
    }
}
