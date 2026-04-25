use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const BASE_URL: &str = "http://localhost:11434";
const TIMEOUT_SECS: u64 = 60;

static CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build()
        .expect("failed to build reqwest client")
});

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Serialize)]
struct GenerateOptions {
    temperature: f32,
    top_p: f32,
    num_predict: i32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

pub async fn generate(model: &str, prompt: &str, temperature: f32) -> Result<String, String> {
    let body = GenerateRequest {
        model,
        prompt,
        stream: false,
        options: GenerateOptions {
            temperature,
            top_p: 0.9,
            num_predict: 32,
        },
    };

    let resp = CLIENT
        .post(format!("{BASE_URL}/api/generate"))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("ollama request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("ollama returned status {}", resp.status()));
    }

    let parsed: GenerateResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse ollama response: {e}"))?;

    Ok(parsed.response)
}

pub async fn ping() -> bool {
    CLIENT
        .get(format!("{BASE_URL}/api/tags"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}
