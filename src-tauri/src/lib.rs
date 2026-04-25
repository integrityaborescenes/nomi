mod backend;

#[tauri::command]
async fn generate_name(
    phrase: String,
    word_count: u8,
    previous: Option<Vec<String>>,
) -> Result<String, String> {
    let prev = previous.unwrap_or_default();
    backend::naming::generate_name(&phrase, word_count, &prev).await
}

#[tauri::command]
async fn check_ollama() -> bool {
    backend::ollama::ping().await
}

#[tauri::command]
async fn check_setup_status() -> backend::setup::SetupStatus {
    backend::setup::check_status().await
}

#[tauri::command]
async fn install_ollama(app: tauri::AppHandle) -> Result<(), String> {
    backend::setup::install_ollama(app).await
}

#[tauri::command]
async fn pull_base_model(app: tauri::AppHandle) -> Result<(), String> {
    backend::setup::pull_base_model(app).await
}

#[tauri::command]
async fn create_custom_model(app: tauri::AppHandle) -> Result<(), String> {
    backend::setup::create_custom_model(app).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            generate_name,
            check_ollama,
            check_setup_status,
            install_ollama,
            pull_base_model,
            create_custom_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
