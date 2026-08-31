//! A sink: POSTs each annotation row arriving on its frames to an HTTP
//! endpoint, one request per row, the row's JSON as the body. Emits no
//! frames and no rows — the requests are its whole product. Imports
//! `wasi:http`, so it runs only under the host's `-http` grant.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, Guest, InFrame, Meta, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;
use wasip2::http::outgoing_handler;
use wasip2::http::types::{Fields, Method, OutgoingBody, OutgoingRequest, Scheme};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"],"additionalProperties":false}"#;

/// The output stream contract: no more than this many bytes per write.
const WRITE_CHUNK: usize = 4096;

#[derive(Deserialize)]
struct Params {
    url: String,
}

/// One parsed destination, split the way an outgoing request wants it.
struct Endpoint {
    https: bool,
    authority: String,
    path: String,
}

thread_local! {
    static ENDPOINT: RefCell<Option<Endpoint>> = const { RefCell::new(None) };
}

fn parse_url(url: &str) -> Result<Endpoint, String> {
    let (https, rest) = if let Some(rest) = url.strip_prefix("http://") {
        (false, rest)
    } else if let Some(rest) = url.strip_prefix("https://") {
        (true, rest)
    } else {
        return Err(format!("url '{url}' is not http:// or https://"));
    };
    let (authority, path) = match rest.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (rest, "/".to_string()),
    };
    if authority.is_empty() {
        return Err(format!("url '{url}' names no host"));
    }
    Ok(Endpoint {
        https,
        authority: authority.to_string(),
        path,
    })
}

fn read_params(params: &str) -> Result<Endpoint, String> {
    let parsed: Params = serde_json::from_str(params)
        .map_err(|e| format!("post_rows takes {{\"url\": \"...\"}}: {e}"))?;
    parse_url(&parsed.url)
}

/// One POST: the row's JSON as the body, judged by the status code alone.
fn post(target: &Endpoint, body: &[u8]) -> Result<(), String> {
    let headers = Fields::new();
    headers
        .append("content-type", b"application/json")
        .map_err(|e| format!("content-type refused: {e:?}"))?;
    headers
        .append("content-length", body.len().to_string().as_bytes())
        .map_err(|e| format!("content-length refused: {e:?}"))?;
    let request = OutgoingRequest::new(headers);
    let scheme = if target.https {
        Scheme::Https
    } else {
        Scheme::Http
    };
    request
        .set_method(&Method::Post)
        .map_err(|()| "POST refused".to_string())?;
    request
        .set_scheme(Some(&scheme))
        .map_err(|()| "scheme refused".to_string())?;
    request
        .set_authority(Some(&target.authority))
        .map_err(|()| format!("authority '{}' refused", target.authority))?;
    request
        .set_path_with_query(Some(&target.path))
        .map_err(|()| format!("path '{}' refused", target.path))?;

    let out_body = request
        .body()
        .map_err(|()| "request body unavailable".to_string())?;
    {
        let stream = out_body
            .write()
            .map_err(|()| "body stream unavailable".to_string())?;
        for chunk in body.chunks(WRITE_CHUNK) {
            stream
                .blocking_write_and_flush(chunk)
                .map_err(|e| format!("writing the body: {e:?}"))?;
        }
    }
    OutgoingBody::finish(out_body, None).map_err(|e| format!("finishing the body: {e:?}"))?;

    let future =
        outgoing_handler::handle(request, None).map_err(|e| format!("request refused: {e:?}"))?;
    future.subscribe().block();
    let response = future
        .get()
        .ok_or_else(|| "response never arrived".to_string())?
        .map_err(|()| "response taken twice".to_string())?
        .map_err(|e| format!("request failed: {e:?}"))?;
    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(format!("endpoint answered {status}"));
    }
    Ok(())
}

struct PostRows;

impl Guest for PostRows {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "post_rows".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                // No rows leave: the requests are the output.
                rows_schema: String::new(),
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // Every call reaches the network.
            pure: false,
            // A sink: frames go in and none come out.
            one_to_one: false,
            // The rows arriving with the frames are what it posts.
            reads_rows: true,
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(_) = format else {
            return Err("post_rows reads pictures, and this stream is audio".to_string());
        };
        let endpoint = read_params(&params)?;
        ENDPOINT.with(|e| *e.borrow_mut() = Some(endpoint));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let endpoint = read_params(&params)?;
        ENDPOINT.with(|e| *e.borrow_mut() = Some(endpoint));
        Ok(())
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, _last: bool) -> Processed {
        ENDPOINT.with(|e| {
            let endpoint_ref = e.borrow();
            let endpoint = endpoint_ref.as_ref().expect("process called before init");

            for frame in &frames {
                for row in &frame.rows {
                    // No error channel on process: a failed POST fails the run.
                    if let Err(message) = post(endpoint, row.as_bytes()) {
                        panic!("post_rows: {message}");
                    }
                }
            }
            for row in &trailing {
                if let Err(message) = post(endpoint, row.as_bytes()) {
                    panic!("post_rows: {message}");
                }
            }

            Processed {
                frames: Vec::new(),
                trailing: Vec::new(),
            }
        })
    }
}

export!(PostRows);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_splits_into_authority_and_path() {
        let endpoint = parse_url("http://127.0.0.1:8123/rows").expect("parses");
        assert!(!endpoint.https);
        assert_eq!(endpoint.authority, "127.0.0.1:8123");
        assert_eq!(endpoint.path, "/rows");
    }

    #[test]
    fn a_bare_host_takes_the_root_path() {
        let endpoint = parse_url("https://example.test").expect("parses");
        assert!(endpoint.https);
        assert_eq!(endpoint.authority, "example.test");
        assert_eq!(endpoint.path, "/");
    }

    #[test]
    fn a_url_without_a_scheme_is_refused() {
        assert!(parse_url("example.test/rows").is_err());
        assert!(parse_url("http://").is_err());
    }
}
