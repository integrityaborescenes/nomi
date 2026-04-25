use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use super::ollama;

const REQUIRED_MODEL: &str = "nomi-namer";
const BASE_MODEL: &str = "qwen2.5:7b-instruct";
const OLLAMA_INSTALLER_URL: &str = "https://ollama.com/download/OllamaSetup.exe";
const MODELFILE_CONTENT: &str = include_str!("../../../Modelfile");

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SetupStatus {
    pub ollama_installed: bool,
    pub ollama_running: bool,
    pub model_ready: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SetupProgress {
    pub stage: String,
    pub percent: Option<u8>,
    pub message: String,
}

pub async fn check_status() -> SetupStatus {
    let ollama_installed = is_ollama_installed().await;
    let ollama_running = ollama::ping().await;
    let model_ready = if ollama_running {
        has_required_model().await
    } else {
        false
    };

    SetupStatus {
        ollama_installed,
        ollama_running,
        model_ready,
    }
}

async fn is_ollama_installed() -> bool {
    Command::new("ollama")
        .arg("--version")
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

async fn has_required_model() -> bool {
    let Ok(tags) = ollama::list_models().await else {
        return false;
    };
    tags.iter()
        .any(|name| name.split(':').next() == Some(REQUIRED_MODEL))
}

fn emit(app: &AppHandle, stage: &str, percent: Option<u8>, message: &str) {
    let _ = app.emit(
        "setup-progress",
        SetupProgress {
            stage: stage.into(),
            percent,
            message: message.into(),
        },
    );
}

pub async fn install_ollama(app: AppHandle) -> Result<(), String> {
    emit(&app, "ollama-download", Some(0), "Скачивание ollama…");

    let installer_path = std::env::temp_dir().join("OllamaSetup.exe");
    download_with_progress(&app, OLLAMA_INSTALLER_URL, &installer_path, "ollama-download")
        .await?;

    emit(&app, "ollama-install", None, "Установка ollama…");
    let status = Command::new(&installer_path)
        .arg("/S")
        .status()
        .await
        .map_err(|e| format!("failed to run installer: {e}"))?;
    if !status.success() {
        return Err(format!("installer exited with {status}"));
    }

    emit(&app, "ollama-install", None, "Ожидание запуска сервиса…");
    for _ in 0..60 {
        if ollama::ping().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Err("ollama did not start within 60 seconds".into())
}

async fn download_with_progress(
    app: &AppHandle,
    url: &str,
    dest: &std::path::Path,
    stage: &str,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("client build failed: {e}"))?;

    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("download returned status {}", response.status()));
    }

    let total = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut last_percent: u8 = 0;

    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("failed to create file: {e}"))?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("chunk read failed: {e}"))?
    {
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write failed: {e}"))?;
        downloaded += chunk.len() as u64;

        if total > 0 {
            let percent = ((downloaded * 100) / total) as u8;
            if percent != last_percent {
                last_percent = percent;
                emit(
                    app,
                    stage,
                    Some(percent),
                    &format!("{} / {} MB", downloaded / 1_048_576, total / 1_048_576),
                );
            }
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("flush failed: {e}"))?;

    Ok(())
}

pub async fn pull_base_model(app: AppHandle) -> Result<(), String> {
    emit(&app, "model-pull", Some(0), "Скачивание модели…");

    let mut child = Command::new("ollama")
        .arg("pull")
        .arg(BASE_MODEL)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn ollama pull: {e}"))?;

    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "no stderr pipe".to_string())?;

    let app_clone = app.clone();
    let reader_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        let percent_re = regex::Regex::new(r"(\d{1,3})%").unwrap();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(cap) = percent_re.captures(&line) {
                if let Ok(p) = cap[1].parse::<u8>() {
                    emit(&app_clone, "model-pull", Some(p), &line);
                }
            }
        }
    });

    let status = child
        .wait()
        .await
        .map_err(|e| format!("ollama pull failed: {e}"))?;
    let _ = reader_handle.await;

    if !status.success() {
        return Err(format!("ollama pull exited with {status}"));
    }

    Ok(())
}

pub async fn create_custom_model(app: AppHandle) -> Result<(), String> {
    emit(&app, "model-create", None, "Подготовка модели…");

    let modelfile_path = std::env::temp_dir().join("nomi-namer.Modelfile");
    tokio::fs::write(&modelfile_path, MODELFILE_CONTENT)
        .await
        .map_err(|e| format!("failed to write Modelfile: {e}"))?;

    let status = Command::new("ollama")
        .arg("create")
        .arg(REQUIRED_MODEL)
        .arg("-f")
        .arg(&modelfile_path)
        .status()
        .await
        .map_err(|e| format!("failed to run ollama create: {e}"))?;

    if !status.success() {
        return Err(format!("ollama create exited with {status}"));
    }

    Ok(())
}
