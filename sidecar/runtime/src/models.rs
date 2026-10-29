//! `ffrwd:models/files` for components that import it: the files bound with
//! `-model <name>=<path>`, handed to a module as bytes.
//!
//! A module that runs its model in its own code reads the weights through
//! this interface. It still has no directory and no path: it names a file by
//! the name it was bound under, and the host reads the file.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Result};

/// The interface a component imports to read its bound files.
pub const IMPORT: &str = "ffrwd:models/files@0.1.0";

/// One `-model` binding: the name a module reads, and the file behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFile {
    pub name: String,
    pub path: PathBuf,
    /// The modules it is bound for, each a `-model-for` after it; empty for
    /// every module.
    pub modules: Vec<PathBuf>,
}

impl ModelFile {
    /// Whether the module at `module` (canonical) reads this binding.
    fn binds(&self, module: &Path) -> bool {
        self.modules.is_empty() || self.modules.iter().any(|m| canonical(m) == module)
    }
}

/// `path` as the file it names, where it names one.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Whether `name` is a plain file name: one path segment, not `.` or `..`.
pub fn is_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\'])
}

/// One `-model` value: `name=path`, with `name` a plain file name and `path`
/// a file that exists now.
pub fn parse(raw: &str) -> Result<ModelFile> {
    let Some((name, path)) = raw.split_once('=') else {
        bail!("-model {raw}: a file is bound as -model name=path");
    };
    if !is_file_name(name) {
        bail!(
            "-model {raw}: '{name}' is not a file name; a name is one path segment, \
             with no / or \\, and not . or .."
        );
    }
    if path.is_empty() {
        bail!("-model {raw}: the name '{name}' is bound to no path");
    }
    let file = Path::new(path);
    if !file.exists() {
        bail!("-model {raw}: '{name}' is bound to {path}, which does not exist");
    }
    if !file.is_file() {
        bail!("-model {raw}: '{name}' is bound to {path}, which is not a file");
    }
    Ok(ModelFile {
        name: name.to_string(),
        path: file.to_path_buf(),
        modules: Vec::new(),
    })
}

/// Refuses a name bound twice for one module: twice for every module, or
/// twice for a module named by two `-model-for`.
pub fn check_bindings(files: &[ModelFile]) -> Result<()> {
    for (at, file) in files.iter().enumerate() {
        for earlier in &files[..at] {
            if earlier.name != file.name {
                continue;
            }
            if earlier.modules.is_empty() && file.modules.is_empty() {
                bail!("-model {}: bound twice", file.name);
            }
            if let Some(module) = file.modules.iter().find(|m| earlier.modules.contains(m)) {
                bail!("-model {}: bound twice for {}", file.name, module.display());
            }
        }
    }
    Ok(())
}

fn bindings() -> &'static OnceLock<Vec<ModelFile>> {
    static FILES: OnceLock<Vec<ModelFile>> = OnceLock::new();
    &FILES
}

/// Records the run's `-model` bindings, once, before any component is
/// instantiated.
pub fn configure(files: Vec<ModelFile>) -> Result<()> {
    check_bindings(&files)?;
    if bindings().set(files).is_err() {
        bail!("the -model bindings are already recorded");
    }
    Ok(())
}

/// The files one module's store reads, by name, in the order they were
/// bound.
#[derive(Clone, Debug, Default)]
pub struct Files(Vec<(String, PathBuf)>);

impl Files {
    /// The bindings of `all` the module at `module_path` reads: its own
    /// binding of a name wins over one for every module.
    fn of(all: &[ModelFile], module_path: &str) -> Files {
        let module = canonical(Path::new(module_path));
        let own = |name: &str| {
            all.iter()
                .any(|f| f.name == name && !f.modules.is_empty() && f.binds(&module))
        };
        Files(
            all.iter()
                .filter(|f| f.binds(&module) && (!f.modules.is_empty() || !own(&f.name)))
                .map(|f| (f.name.clone(), f.path.clone()))
                .collect(),
        )
    }

    /// What the run bound for the module at `module_path`.
    pub fn bound_for(module_path: &str) -> Files {
        match bindings().get() {
            Some(all) => Files::of(all, module_path),
            None => Files::default(),
        }
    }

    /// The bound names, in the order they were bound.
    pub fn names(&self) -> Vec<String> {
        self.0.iter().map(|(name, _)| name.clone()).collect()
    }

    /// The whole file bound as `name`, read now.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let Some((_, path)) = self.0.iter().find(|(n, _)| n == name) else {
            if self.0.is_empty() {
                return Err(format!(
                    "no file named '{name}' is bound for this module; no file is bound for it"
                ));
            }
            return Err(format!(
                "no file named '{name}' is bound for this module; the files bound for it \
                 are {}",
                self.names().join(", ")
            ));
        };
        std::fs::read(path).map_err(|e| format!("'{name}' could not be read: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(name: &str, path: &str, modules: &[&str]) -> ModelFile {
        ModelFile {
            name: name.to_string(),
            path: path.into(),
            modules: modules.iter().map(PathBuf::from).collect(),
        }
    }

    #[test]
    fn a_module_reads_its_own_binding_over_one_for_every_module_in_argv_order() {
        let all = [
            bound("w.bin", "every/w.bin", &[]),
            bound("cfg.json", "every/cfg.json", &[]),
            bound("w.bin", "a/w.bin", &["a.wasm"]),
            bound("x.bin", "b/x.bin", &["b.wasm"]),
        ];
        let a = Files::of(&all, "a.wasm");
        assert_eq!(
            a.0,
            vec![
                ("cfg.json".to_string(), PathBuf::from("every/cfg.json")),
                ("w.bin".to_string(), PathBuf::from("a/w.bin")),
            ]
        );
        let b = Files::of(&all, "b.wasm");
        assert_eq!(b.names(), ["w.bin", "cfg.json", "x.bin"]);
    }

    #[test]
    fn an_unbound_name_is_refused_naming_what_is_bound() {
        let all = [bound("w.bin", "w.bin", &[]), bound("c.json", "c.json", &[])];
        assert_eq!(
            Files::of(&all, "m.wasm").read("nope.bin").unwrap_err(),
            "no file named 'nope.bin' is bound for this module; the files bound for it are \
             w.bin, c.json"
        );
        assert_eq!(
            Files::default().read("nope.bin").unwrap_err(),
            "no file named 'nope.bin' is bound for this module; no file is bound for it"
        );
    }

    #[test]
    fn file_names_are_one_segment() {
        for good in ["w.bin", "model.onnx", "a..b", ".hidden"] {
            assert!(is_file_name(good), "{good}");
        }
        for bad in ["", ".", "..", "a/b", "a\\b", "/w.bin"] {
            assert!(!is_file_name(bad), "{bad}");
        }
    }
}
