import styles from "./WordCountSelector.module.scss";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../../store/store.ts";
import { setCount } from "../../store/slices/parametersSlice.ts";

const WordCountSelector = () => {
  const dispatch = useDispatch<AppDispatch>();
  const wordCounts = [2, 3, 4, 5];
  const [selectedId, setSelectedId] = useState(0);

  return (
    <div
      className={styles.wordCountSelector}
      style={
        {
          "--i": selectedId,
          "--count": wordCounts.length,
        } as React.CSSProperties
      }
    >
      {wordCounts.map((c, id) => (
        <button
          key={c}
          className={selectedId === id ? styles.selected : ""}
          onClick={() => {
            setSelectedId(id);
            dispatch(setCount(c));
          }}
        >
          {c}
        </button>
      ))}
    </div>
  );
};

export default WordCountSelector;
