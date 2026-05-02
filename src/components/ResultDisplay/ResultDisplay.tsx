import styles from "./ResultDisplay.module.scss";
import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store.ts";
import { invoke } from "@tauri-apps/api/core";

const ResultDisplay = () => {
  const wordCount = useSelector(
    (state: RootState) => state.parametersSlice.countOfWords,
  );

  const phrase = useSelector(
    (state: RootState) => state.parametersSlice.inputText,
  );

  const submitNonce = useSelector(
    (state: RootState) => state.parametersSlice.submitNonce,
  );

  const [result, setResult] = useState<string | null>(null);
  const [spinning, setSpinning] = useState(false);
  const [copied, setCopied] = useState(false);
  const [errorGenerate, setErrorGenerate] = useState<boolean>(false);
  const [retry, setRetry] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  const lastKeyRef = useRef<string>("");

  const handleRetry = () => {
    if (loading) return;
    if (!result && !errorGenerate) return;

    setSpinning(true);
    setRetry((prev) => prev + 1);
  };

  const handleCopy = async () => {
    if (!result) return;

    await navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 1000);
  };

  useEffect(() => {
    if (!phrase) return;

    const key = `${phrase}:${wordCount}`;
    const sameKey = lastKeyRef.current === key;
    lastKeyRef.current = key;
    const previous = sameKey && result ? [result] : [];

    (async () => {
      setLoading(true);
      setErrorGenerate(false);
      try {
        const name = await invoke<string>("generate_name", {
          phrase,
          wordCount,
          previous,
        });
        setResult(name);
      } catch (e) {
        console.error("generate_name failed:", e);
        setErrorGenerate(true);
      } finally {
        setLoading(false);
      }
    })();
  }, [submitNonce, retry]);

  return (
    <div className={styles.resultDisplay}>
      {loading ? (
        <span className={styles.loader} />
      ) : (
        <p>
          {errorGenerate ? "Не получилось :( попробуй ещё раз" : result}
        </p>
      )}
      <div className={styles.controls}>
        <button
          className={`${styles.retry} ${spinning ? styles.spinning : ""}`}
          onClick={handleRetry}
          onAnimationEnd={() => setSpinning(false)}
        >
          <img draggable={false} src="/icons/retryIco.svg" />
        </button>
        <button className={styles.copy} onClick={handleCopy}>
          {copied ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
              <path
                d="M5 12l5 5L20 7"
                stroke="#6B7280"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <img draggable={false} src="/icons/copyIco.svg" />
          )}
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay;
