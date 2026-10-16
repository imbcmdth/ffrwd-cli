//! Which adapter the process's device is opened on: the list `--gpu-info`
//! prints, the one wgpu would pick, and the one a `-gpu-adapter` spec (or
//! `FFRWD_GPU_ADAPTER`) names.
//!
//! The list is every adapter of each preferred backend, the backends in
//! their preferred order and each backend's adapters in the order it
//! reports them. A spec is a zero-based index into it, or a case-insensitive
//! substring of an adapter's name; digits past the end of the list are read
//! as a name, so `4090` names a card. A name is looked for backend by backend:
//! the first backend with an adapter so named answers, so `4090` is the
//! D3D12 adapter on Windows even where Vulkan reports the same card, and a
//! backend with several so named refuses it.

use std::fmt;

/// The environment variable naming the adapter when the argv does not.
pub const ADAPTER_VAR: &str = "FFRWD_GPU_ADAPTER";

/// The flag naming the adapter.
pub const ADAPTER_FLAG: &str = "-gpu-adapter";

/// Where the full list is printed, named in every refusal.
const INFO_HINT: &str = "`ffrwd-wasm --gpu-info` lists them";

/// What kind of adapter it is, as `--gpu-info` spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Discrete,
    Integrated,
    Virtual,
    Software,
    Other,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Discrete => "discrete",
            Kind::Integrated => "integrated",
            Kind::Virtual => "virtual",
            Kind::Software => "software",
            Kind::Other => "other",
        }
    }

    /// wgpu's order for a high-performance request: discrete first, then
    /// integrated, other, virtual, software.
    fn rank(self) -> u8 {
        match self {
            Kind::Discrete => 1,
            Kind::Integrated => 2,
            Kind::Other => 3,
            Kind::Virtual => 4,
            Kind::Software => 5,
        }
    }
}

/// One adapter of the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listed {
    pub name: String,
    /// wgpu's name for the backend: `dx12`, `vulkan`, `metal`, `gl`.
    pub backend: &'static str,
    pub vendor: u32,
    pub device: u32,
    pub kind: Kind,
}

/// The index wgpu's own high-performance request picks: in the first
/// backend that has any adapter, the first of the best kind.
pub fn default_index(list: &[Listed]) -> Option<usize> {
    let first = list.first()?.backend;
    list.iter()
        .enumerate()
        .filter(|(_, one)| one.backend == first)
        .min_by_key(|(_, one)| one.kind.rank())
        .map(|(index, _)| index)
}

/// A spec, and where it was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub raw: String,
    pub from_env: bool,
}

impl Spec {
    /// The flag's value, which replaces the environment's; else the
    /// environment's, where it is set and not blank.
    pub fn given(flag: Option<String>) -> Option<Spec> {
        if let Some(raw) = flag {
            return Some(Spec {
                raw,
                from_env: false,
            });
        }
        let raw = std::env::var_os(ADAPTER_VAR)?.to_string_lossy().to_string();
        if raw.trim().is_empty() {
            return None;
        }
        Some(Spec {
            raw,
            from_env: true,
        })
    }

    /// How a refusal names it: `-gpu-adapter radeon` or
    /// `FFRWD_GPU_ADAPTER=radeon`.
    fn label(&self) -> String {
        if self.from_env {
            format!("{ADAPTER_VAR}={}", self.raw)
        } else {
            format!("{ADAPTER_FLAG} {}", self.raw)
        }
    }
}

/// Why a spec names no one adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Why {
    /// The spec is blank.
    Blank,
    /// The machine has no adapter at all.
    NoAdapters,
    /// Digits past the end of the list, which no adapter's name contains
    /// either.
    OutOfRange,
    /// No adapter's name contains it.
    NoMatch,
    /// Several adapters of one backend have it in their names, at these
    /// indices.
    Ambiguous(Vec<usize>),
}

/// A spec refused, with the list it was matched against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub spec: Spec,
    pub why: Why,
    pub adapters: Vec<Listed>,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.spec.label();
        let raw = self.spec.raw.trim();
        match &self.why {
            Why::Blank => return write!(f, "{label}: names no adapter; {INFO_HINT}"),
            Why::NoAdapters => return write!(f, "{label}: this machine has no GPU adapter"),
            Why::OutOfRange => write!(
                f,
                "{label}: there is no adapter {raw}, they are numbered from 0 to {}, and no \
                 adapter's name contains '{raw}'",
                self.adapters.len() - 1
            )?,
            Why::NoMatch => write!(f, "{label}: no adapter's name contains '{raw}'")?,
            Why::Ambiguous(indices) => {
                let backend = self.adapters[indices[0]].backend;
                write!(
                    f,
                    "{label}: {} {backend} adapters' names contain '{raw}'; name one by its \
                     index",
                    indices.len()
                )?
            }
        }
        let listed: Vec<String> = self
            .adapters
            .iter()
            .enumerate()
            .map(|(index, one)| format!("{index} {} ({})", one.name, one.backend))
            .collect();
        write!(f, "; the adapters are {}; {INFO_HINT}", listed.join(", "))
    }
}

impl std::error::Error for Refusal {}

/// The index of the adapter `spec` names in `list`.
pub fn resolve(spec: &Spec, list: &[Listed]) -> Result<usize, Refusal> {
    let refuse = |why: Why| Refusal {
        spec: spec.clone(),
        why,
        adapters: list.to_vec(),
    };
    let raw = spec.raw.trim();
    if raw.is_empty() {
        return Err(refuse(Why::Blank));
    }
    if list.is_empty() {
        return Err(refuse(Why::NoAdapters));
    }
    let digits = raw.bytes().all(|b| b.is_ascii_digit());
    if digits {
        if let Ok(index) = raw.parse::<usize>() {
            if index < list.len() {
                return Ok(index);
            }
        }
    }
    let wanted = raw.to_lowercase();
    let named: Vec<usize> = list
        .iter()
        .enumerate()
        .filter(|(_, one)| one.name.to_lowercase().contains(&wanted))
        .map(|(index, _)| index)
        .collect();
    let Some(&first) = named.first() else {
        return Err(refuse(if digits {
            Why::OutOfRange
        } else {
            Why::NoMatch
        }));
    };
    let backend = list[first].backend;
    let same: Vec<usize> = named
        .into_iter()
        .filter(|&index| list[index].backend == backend)
        .collect();
    match same.as_slice() {
        [one] => Ok(*one),
        _ => Err(refuse(Why::Ambiguous(same))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, backend: &'static str, kind: Kind) -> Listed {
        Listed {
            name: name.to_string(),
            backend,
            vendor: 0,
            device: 0,
            kind,
        }
    }

    fn machine() -> Vec<Listed> {
        vec![
            adapter("Microsoft Basic Render Driver", "dx12", Kind::Software),
            adapter("AMD Radeon(TM) Graphics", "dx12", Kind::Integrated),
            adapter("NVIDIA GeForce RTX 4090", "dx12", Kind::Discrete),
            adapter("NVIDIA GeForce RTX 4090", "vulkan", Kind::Discrete),
            adapter("llvmpipe (LLVM 17.0.6, 256 bits)", "vulkan", Kind::Software),
        ]
    }

    fn flag(raw: &str) -> Spec {
        Spec {
            raw: raw.to_string(),
            from_env: false,
        }
    }

    #[test]
    fn the_default_is_the_best_kind_of_the_first_backend() {
        assert_eq!(default_index(&machine()), Some(2));
        assert_eq!(default_index(&machine()[3..]), Some(0));
        assert_eq!(default_index(&machine()[..2]), Some(1));
        assert_eq!(default_index(&[]), None);
    }

    #[test]
    fn a_spec_is_an_index_or_a_name_looked_for_backend_by_backend() {
        let list = machine();
        for (raw, index) in [
            ("0", 0),
            ("3", 3),
            (" 4 ", 4),
            ("radeon", 1),
            ("4090", 2),
            ("17", 4),
            ("RTX", 2),
            ("Basic Render", 0),
            ("llvmpipe", 4),
        ] {
            assert_eq!(resolve(&flag(raw), &list), Ok(index), "{raw}");
        }
    }

    #[test]
    fn a_spec_naming_no_one_adapter_is_refused_with_the_list() {
        let list = machine();
        let names = "the adapters are 0 Microsoft Basic Render Driver (dx12), \
             1 AMD Radeon(TM) Graphics (dx12), 2 NVIDIA GeForce RTX 4090 (dx12), \
             3 NVIDIA GeForce RTX 4090 (vulkan), 4 llvmpipe (LLVM 17.0.6, 256 bits) (vulkan); \
             `ffrwd-wasm --gpu-info` lists them";
        let env = Spec {
            raw: "intel".to_string(),
            from_env: true,
        };
        for (spec, list, message) in [
            (
                flag("arc"),
                &list[..],
                format!("-gpu-adapter arc: no adapter's name contains 'arc'; {names}"),
            ),
            (
                env,
                &list[..],
                format!("FFRWD_GPU_ADAPTER=intel: no adapter's name contains 'intel'; {names}"),
            ),
            (
                flag("8"),
                &list[..],
                format!(
                    "-gpu-adapter 8: there is no adapter 8, they are numbered from 0 to 4, and \
                     no adapter's name contains '8'; {names}"
                ),
            ),
            (
                flag("99999999999999999999999"),
                &list[..],
                format!(
                    "-gpu-adapter 99999999999999999999999: there is no adapter \
                     99999999999999999999999, they are numbered from 0 to 4, and no adapter's \
                     name contains '99999999999999999999999'; {names}"
                ),
            ),
            (
                flag("geforce"),
                &[list[2].clone(), list[2].clone(), list[3].clone()][..],
                "-gpu-adapter geforce: 2 dx12 adapters' names contain 'geforce'; name one \
                 by its index; the adapters are 0 NVIDIA GeForce RTX 4090 (dx12), \
                 1 NVIDIA GeForce RTX 4090 (dx12), 2 NVIDIA GeForce RTX 4090 (vulkan); \
                 `ffrwd-wasm --gpu-info` lists them"
                    .to_string(),
            ),
            (
                flag(" "),
                &list[..],
                "-gpu-adapter  : names no adapter; `ffrwd-wasm --gpu-info` lists them".to_string(),
            ),
            (
                flag("0"),
                &[][..],
                "-gpu-adapter 0: this machine has no GPU adapter".to_string(),
            ),
        ] {
            let refused = resolve(&spec, list).expect_err(&spec.raw);
            assert_eq!(refused.to_string(), message);
        }
    }
}
