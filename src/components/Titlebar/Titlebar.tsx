import { useState } from "react";
import styles from "./Titlebar.module.scss";
import { getCurrentWindow } from "@tauri-apps/api/window";

const Titlebar = () => {
  const win = getCurrentWindow();
  const [pinned, setPinned] = useState(false);

  const togglePin = async () => {
    const next = !pinned;
    await win.setAlwaysOnTop(next);
    setPinned(next);
  };

  return (
    <div className={styles.titlebar} data-tauri-drag-region>
      <span className={styles.title} data-tauri-drag-region>
        nomi
      </span>
      <div className={styles.controls}>
        <button
          className={`${styles.pin} ${pinned ? styles.pinActive : ""}`}
          onClick={togglePin}
          aria-label={pinned ? "unpin from top" : "pin on top"}
          title={pinned ? "Открепить" : "Закрепить поверх всех окон"}
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill={pinned ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="12" y1="17" x2="12" y2="22" />
            <path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z" />
          </svg>
        </button>
        <button
          className={styles.minimize}
          onClick={() => win.minimize()}
          aria-label="minimize"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <line
              x1="2"
              y1="5"
              x2="8"
              y2="5"
              stroke="currentColor"
              strokeWidth="1"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <button
          className={styles.close}
          onClick={() => win.close()}
          aria-label="close"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <line
              x1="2"
              y1="2"
              x2="8"
              y2="8"
              stroke="currentColor"
              strokeWidth="1"
              strokeLinecap="round"
            />
            <line
              x1="8"
              y1="2"
              x2="2"
              y2="8"
              stroke="currentColor"
              strokeWidth="1"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default Titlebar;
