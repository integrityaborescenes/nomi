import styles from "./Titlebar.module.scss";
import { getCurrentWindow } from "@tauri-apps/api/window";

const Titlebar = () => {
  const win = getCurrentWindow();

  return (
    <div className={styles.titlebar} data-tauri-drag-region>
      <span className={styles.title} data-tauri-drag-region>
        nomi
      </span>
      <div className={styles.controls}>
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
