import styles from "./ResultDisplay.module.scss";
import { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store.ts";
import { invoke } from "@tauri-apps/api/core";

type NameResult = {
  pascal: string;
  camel: string;
  kebab: string;
  snake: string;
};

const ResultDisplay = () => {
  const style = useSelector((state: RootState) => state.parametersSlice.style);

  const phrase = useSelector(
    (state: RootState) => state.parametersSlice.inputText,
  );

  const submitNonce = useSelector(
    (state: RootState) => state.parametersSlice.submitNonce,
  );

  const [result, setResult] = useState<NameResult | null>(null);
  const [spinning, setSpinning] = useState(false);
  const [copied, setCopied] = useState(false);
  const [tooltipSuppressed, setTooltipSuppressed] = useState(false);
  const [errorGenerate, setErrorGenerate] = useState<boolean>(false);
  const [retry, setRetry] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingText, setLoadingText] = useState<string>("");

  const lastPhraseRef = useRef<string>("");

  const display = result ? result[style] : null;

  const handleRetry = () => {
    if (loading) return;
    if (!result && !errorGenerate) return;

    setSpinning(true);
    setRetry((prev) => prev + 1);
  };

  const handleCopy = async () => {
    if (!display) return;

    await navigator.clipboard.writeText(display);
    setCopied(true);
    setTooltipSuppressed(true);
    setTimeout(() => setCopied(false), 1000);
  };

  useEffect(() => {
    if (!phrase) return;

    const samePhrase = lastPhraseRef.current === phrase;
    lastPhraseRef.current = phrase;
    const previous = samePhrase && result ? [result.pascal] : [];

    (async () => {
      setLoading(true);
      setErrorGenerate(false);
      try {
        const ready = await invoke<boolean>("is_model_ready");
        setLoadingText(ready ? "" : "Загрузка модели");
        const name = await invoke<NameResult>("generate_name", {
          phrase,
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
        <div className={styles.loadingRow}>
          <span className={styles.loader} />
          <span className={styles.loadingText}>{loadingText}</span>
        </div>
      ) : (
        <p
          onClick={display ? handleCopy : undefined}
          onMouseLeave={() => setTooltipSuppressed(false)}
          className={display ? styles.clickable : undefined}
          data-tooltip={
            !display
              ? undefined
              : copied
                ? "Скопировано"
                : tooltipSuppressed
                  ? undefined
                  : "Скопировать"
          }
        >
          {errorGenerate ? "Не получилось :( попробуй ещё раз" : display}
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
