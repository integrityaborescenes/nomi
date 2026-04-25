import styles from "./SetupWizard.module.scss";
import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

type SetupStatus = {
  ollamaInstalled: boolean;
  ollamaRunning: boolean;
  modelReady: boolean;
};

type Progress = {
  stage: string;
  percent: number | null;
  message: string;
};

type Props = {
  onReady: () => void;
};

const SetupWizard = ({ onReady }: Props) => {
  const [progress, setProgress] = useState<Progress>({
    stage: "init",
    percent: null,
    message: "Проверка…",
  });
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    let unlisten: UnlistenFn | null = null;
    let cancelled = false;

    (async () => {
      unlisten = await listen<Progress>("setup-progress", (e) => {
        setProgress(e.payload);
      });

      if (startedRef.current) return;
      startedRef.current = true;

      try {
        const status = await invoke<SetupStatus>("check_setup_status");

        if (!status.ollamaInstalled) {
          await invoke("install_ollama");
        }

        if (!status.modelReady) {
          const after = await invoke<SetupStatus>("check_setup_status");
          if (!after.modelReady) {
            await invoke("pull_base_model");
            await invoke("create_custom_model");
          }
        }

        if (!cancelled) onReady();
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();

    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [onReady]);

  return (
    <div className={styles.wizard}>
      {error ? (
        <p className={styles.error}>{error}</p>
      ) : (
        <>
          <p className={styles.message}>{progress.message}</p>
          <div className={styles.barTrack}>
            {progress.percent != null ? (
              <div
                className={styles.barFill}
                style={{ width: `${progress.percent}%` }}
              />
            ) : (
              <div className={`${styles.barFill} ${styles.barIndeterminate}`} />
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default SetupWizard;
