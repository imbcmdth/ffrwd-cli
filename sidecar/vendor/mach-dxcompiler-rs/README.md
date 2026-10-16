# mach-dxcompiler-rs (vendored)

The crate that downloads a prebuilt DXC and links it statically. wgpu-hal
pulls it in through `static-dxc`, which the sidecar enables on x86_64 MSVC
Windows (see `../wasi-webgpu-wasmtime/Cargo.toml`), so it builds only for that
target.

Taken from crates.io, mach-dxcompiler-rs 0.1.6+2026.09.16-48d5a66.1 (upstream
[DouglasDwyer/mach-dxcompiler-rs](https://github.com/DouglasDwyer/mach-dxcompiler-rs)
at 77b9a2f), the version `../../Cargo.lock` resolved. MIT, as upstream
(`LICENSE`). This file replaces upstream's README, which describes the crate's
use.

## Why

Before it downloads anything, the build script asks the GitHub API whether the
release it downloads from is immutable, and refuses to build unless the answer
says so. Upstream makes that request without authentication. On a shared
GitHub Actions runner the unauthenticated rate limit is often used up, the
answer is a rate-limit error with no `immutable` field, and the build fails
with "Refusing to download a mutable GitHub release" although the release is
immutable. That failed a release's Windows x86_64 wheel build.

## What changed from upstream

* `build.rs`: when `GITHUB_TOKEN` is set and not empty, the release-metadata
  request sends `Authorization: Bearer <token>`. curl reads the header from
  stdin, so the token is not on its command line. The archive download that
  follows is unchanged and sends no token: it is redirected to GitHub's asset
  storage, which needs none. The immutability check and the archive's SHA-256
  check are as upstream.
* This README.

The release workflows set `GITHUB_TOKEN` to the workflow's own token for the
steps that build the sidecar on Windows. When upstream authenticates this
request itself, this copy can go.
