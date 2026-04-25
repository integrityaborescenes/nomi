import styles from "./ResultDisplay.module.scss";
import { useState } from "react";

const ResultDisplay = () => {
  const result = "UserDashboardHeader";

  const [spinning, setSpinning] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleRetry = () => {
    setSpinning(true);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 1000);
  };

  return (
    <div className={styles.resultDisplay}>
      <p>{result}</p>
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
