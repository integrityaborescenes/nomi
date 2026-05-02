mod backend;

#[tauri::command]
async fn generate_name(
    phrase: String,
    word_count: u8,
    previous: Option<Vec<String>>,
) -> Result<String, String> {
    let prev = previous.unwrap_or_default();
    tokio::task::spawn_blocking(move || {
        backend::inference::generate_name(&phrase, word_count, &prev)
    })
    .await
    .map_err(|e| format!("join: {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![generate_name])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
