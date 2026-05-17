import styles from "./PhraseInput.module.scss";
import { ChangeEvent, KeyboardEvent, useState } from "react";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../../store/store.ts";
import { setInputText, bumpSubmit } from "../../store/slices/parametersSlice.ts";

const PhraseInput = () => {
  const dispatch = useDispatch<AppDispatch>();

  const [inputValue, setInputValue] = useState<string>("");
  const handleInput = (e: ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleSetInput = () => {
    if (inputValue === "") return;
    dispatch(setInputText(inputValue));
    dispatch(bumpSubmit());
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSetInput();
    }
  };

  return (
    <div className={styles.phaseInput}>
      <input
        type={"text"}
        placeholder={"Describe a component…"}
        value={inputValue}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
      />
      <button
        className={styles.sendButton}
        onClick={handleSetInput}
        disabled={inputValue === ""}
      >
        <img draggable={false} src="/icons/arrowTopIco.svg" />
      </button>
    </div>
  );
};

export default PhraseInput;
