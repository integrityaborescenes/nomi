mod backend;

use backend::inference::NameResult;
use tauri::Manager;

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
        .setup(|app| {
            // Прогрев модели в фоне — пока юзер набирает фразу,
            // 145MB веса уже грузятся в RAM. Первая генерация моментальная.
            std::thread::spawn(|| backend::inference::warmup());

            // Глобальный хоткей Ctrl+N — показать/скрыть окно.
            #[cfg(desktop)]
            {
                use tauri_plugin_global_shortcut::{
                    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
                };

                let ctrl_n = Shortcut::new(Some(Modifiers::CONTROL), Code::KeyN);
                app.handle().plugin(
                    tauri_plugin_global_shortcut::Builder::new()
                        .with_handler(move |app, shortcut, event| {
                            if event.state() == ShortcutState::Pressed && shortcut == &ctrl_n {
                                if let Some(window) = app.get_webview_window("main") {
                                    if window.is_visible().unwrap_or(false) {
                                        let _ = window.hide();
                                    } else {
                                        let _ = window.show();
                                        let _ = window.set_focus();
                                    }
                                }
                            }
                        })
                        .build(),
                )?;
                app.global_shortcut().register(ctrl_n)?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![generate_name, is_model_ready])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
