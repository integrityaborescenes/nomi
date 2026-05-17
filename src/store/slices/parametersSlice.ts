import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

export type NameStyle = "pascal" | "camel" | "kebab";

type ParametersState = {
  style: NameStyle;
  inputText: string | null;
  submitNonce: number;
};

const initialState: ParametersState = {
  style: "pascal",
  inputText: null,
  submitNonce: 0,
};

export const parametersSlice = createSlice({
  name: "parameters",
  initialState,
  reducers: {
    setStyle: (state, action: PayloadAction<NameStyle>) => {
      state.style = action.payload;
    },
    setInputText: (state, action: PayloadAction<string | null>) => {
      state.inputText = action.payload;
    },
    bumpSubmit: (state) => {
      state.submitNonce += 1;
    },
    clearParameters: () => initialState,
  },
});

export const { setStyle, setInputText, bumpSubmit, clearParameters } = parametersSlice.actions;
export default parametersSlice.reducer;
