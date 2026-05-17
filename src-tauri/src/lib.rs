mod backend;

use backend::inference::NameResult;

#[tauri::command]
async fn generate_name(
    phrase: String,
    previous: Option<Vec<String>>,
) -> Result<NameResult, String> {
    let prev = previous.unwrap_or_default();
    tokio::task::spawn_blocking(move || backend::inference::generate_name(&phrase, &prev))
        .await
        .map_err(|e| format!("join: {e}"))?
}

#[tauri::command]
fn is_model_ready() -> bool {
    backend::inference::is_warmed_up()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|_app| {
            // Прогрев модели в фоне — пока юзер набирает фразу,
            // 145MB веса уже грузятся в RAM. Первая генерация моментальная.
            std::thread::spawn(|| backend::inference::warmup());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![generate_name, is_model_ready])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
