use std::fmt;
use std::str::FromStr;

use yuzu_types::{AggFunc, Func};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    DataFusion,
    Postgres,
}

impl fmt::Display for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dialect::DataFusion => "datafusion",
            Dialect::Postgres => "postgres",
        };
        f.write_str(name)
    }
}

/// Where the plan will run: a dialect, optionally pinned to a version.
/// Without a version the newest capabilities are assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub dialect: Dialect,
    pub version: Option<u32>,
}

impl Target {
    pub fn supports(&self, support: Support) -> bool {
        match support {
            Support::Yes => true,
            Support::No => false,
            Support::Since(version) => self.version.is_none_or(|target| target >= version),
        }
    }

    pub fn registry(&self) -> Box<dyn TargetRegistry> {
        match self.dialect {
            Dialect::DataFusion => Box::new(DataFusionTarget),
            Dialect::Postgres => Box::new(PostgresTarget),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            Some(version) => write!(f, "{} {version}", self.dialect),
            None => write!(f, "{}", self.dialect),
        }
    }
}

impl FromStr for Target {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (dialect, version) = match text.split_once('@') {
            Some((dialect, version)) => {
                let version = version
                    .parse()
                    .map_err(|_| format!("`{version}` is not a version number"))?;
                (dialect, Some(version))
            }
            None => (text, None),
        };
        let dialect = match dialect {
            "datafusion" => Dialect::DataFusion,
            "postgres" => Dialect::Postgres,
            other => return Err(format!("`{other}` is not a known target")),
        };
        Ok(Target { dialect, version })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    Yes,
    No,
    Since(u32),
}

/// What one dialect can execute. Consulted when validating a plan graph
/// against a target; emitters stay free to spell the supported functions
/// however the dialect requires.
pub trait TargetRegistry {
    fn scalar(&self, func: Func) -> Support;
    fn aggregate(&self, func: AggFunc) -> Support;
}

struct DataFusionTarget;

impl TargetRegistry for DataFusionTarget {
    fn scalar(&self, func: Func) -> Support {
        match func {
            // DataFusion registers no `shift_left`/`shift_right`, and `**` has
            // no Substrait mapping yet.
            Func::ShiftLeft | Func::ShiftRight | Func::Power => Support::No,
            // An external declaration is the user's promise the target has it.
            Func::External(_) => Support::Yes,
            _ => Support::Yes,
        }
    }

    fn aggregate(&self, _func: AggFunc) -> Support {
        Support::Yes
    }
}

struct PostgresTarget;

impl TargetRegistry for PostgresTarget {
    fn scalar(&self, func: Func) -> Support {
        match func {
            // No Substrait mapping yet, on any target.
            Func::Power => Support::No,
            _ => Support::Yes,
        }
    }

    fn aggregate(&self, _func: AggFunc) -> Support {
        Support::Yes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_targets() {
        assert_eq!(
            "datafusion".parse::<Target>().unwrap(),
            Target {
                dialect: Dialect::DataFusion,
                version: None
            }
        );
        assert_eq!(
            "postgres@16".parse::<Target>().unwrap(),
            Target {
                dialect: Dialect::Postgres,
                version: Some(16)
            }
        );
        assert!("postgres@new".parse::<Target>().is_err());
        assert!("mysql".parse::<Target>().is_err());
    }

    #[test]
    fn since_resolves_against_the_version() {
        let old = Target {
            dialect: Dialect::Postgres,
            version: Some(15),
        };
        let new = Target {
            dialect: Dialect::Postgres,
            version: Some(16),
        };
        let latest = Target {
            dialect: Dialect::Postgres,
            version: None,
        };
        assert!(!old.supports(Support::Since(16)));
        assert!(new.supports(Support::Since(16)));
        assert!(latest.supports(Support::Since(16)));
        assert!(old.supports(Support::Yes));
        assert!(!latest.supports(Support::No));
    }

    #[test]
    fn datafusion_rejects_shifts() {
        let target = Target {
            dialect: Dialect::DataFusion,
            version: None,
        };
        let registry = target.registry();
        assert!(!target.supports(registry.scalar(Func::ShiftLeft)));
        assert!(target.supports(registry.scalar(Func::Add)));
        assert!(target.supports(registry.aggregate(AggFunc::CountDistinct)));
    }

    #[test]
    fn postgres_accepts_shifts() {
        let target = Target {
            dialect: Dialect::Postgres,
            version: None,
        };
        let registry = target.registry();
        assert!(target.supports(registry.scalar(Func::ShiftLeft)));
    }
}
