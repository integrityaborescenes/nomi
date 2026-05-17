import styles from "./StyleSelector.module.scss";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../../store/store.ts";
import { setStyle, type NameStyle } from "../../store/slices/parametersSlice.ts";

const options: { id: NameStyle; label: string }[] = [
  { id: "pascal", label: "Pascal" },
  { id: "camel", label: "camel" },
  { id: "kebab", label: "kebab" },
  { id: "snake", label: "snake" },
];

const StyleSelector = () => {
  const dispatch = useDispatch<AppDispatch>();
  const [selectedId, setSelectedId] = useState(0);

  return (
    <div
      className={styles.styleSelector}
      style={
        {
          "--i": selectedId,
          "--count": options.length,
        } as React.CSSProperties
      }
    >
      {options.map((opt, id) => (
        <button
          key={opt.id}
          className={selectedId === id ? styles.selected : ""}
          onClick={() => {
            setSelectedId(id);
            dispatch(setStyle(opt.id));
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
};

export default StyleSelector;
