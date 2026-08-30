//! Finds the whisper files the module compiles in, and tells the crate whether
//! they were there.
//!
//! The weights are too big to keep in git, so a checkout may not have them.
//! Rather than fail the build - the modules workspace is built whole on
//! machines that will never run this module - the crate is built without them
//! and refuses at init instead. `fetch-model.ps1` puts them where this looks.

use std::path::PathBuf;

/// The files that must all be present for the model to be compiled in.
const FILES: [&str; 4] = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
    "melfilters.bytes",
];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(have_model)");
    println!("cargo::rerun-if-env-changed=TRANSCRIBE_MODEL_DIR");

    let Ok(dir) = std::env::var("TRANSCRIBE_MODEL_DIR") else {
        println!(
            "cargo::warning=TRANSCRIBE_MODEL_DIR is unset; transcribe is built without its model"
        );
        return;
    };
    let dir = PathBuf::from(dir);

    let paths: Vec<PathBuf> = FILES.iter().map(|f| dir.join(f)).collect();
    for path in &paths {
        println!("cargo::rerun-if-changed={}", path.display());
    }

    let missing: Vec<&str> = FILES
        .iter()
        .zip(&paths)
        .filter(|(_, path)| !path.is_file())
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        println!(
            "cargo::warning=transcribe is built without its model: {} missing from {}",
            missing.join(", "),
            dir.display()
        );
        return;
    }

    for (name, path) in FILES.iter().zip(&paths) {
        // WEIGHTS, TOKENIZER, CONFIG, MELFILTERS.
        let key = name
            .split('.')
            .next()
            .expect("a file name has a first piece")
            .to_uppercase();
        let key = if key == "MODEL" { "WEIGHTS" } else { &key };
        println!("cargo::rustc-env=TRANSCRIBE_{key}={}", path.display());
    }
    println!("cargo::rustc-cfg=have_model");
}
