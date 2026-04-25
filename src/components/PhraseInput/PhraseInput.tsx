import styles from "./PhraseInput.module.scss";

const PhraseInput = () => {
  return (
    <div className={styles.phaseInput}>
      <input type={"text"} placeholder={"Describe a component…"} />
      <div className={styles.sendButton}>
        <img draggable={false} src="/icons/arrowTopIco.svg" />
      </div>
    </div>
  );
};

export default PhraseInput;
