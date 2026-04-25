mod backend;

#[tauri::command]
async fn generate_name(phrase: String, word_count: u8) -> Result<String, String> {
    backend::naming::generate_name(&phrase, word_count).await
}

#[tauri::command]
async fn check_ollama() -> bool {
    backend::ollama::ping().await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![generate_name, check_ollama])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
