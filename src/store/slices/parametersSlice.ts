import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

type ParametersState = {
  countOfWords: number;
  inputText: string | null;
  submitNonce: number;
};

const initialState: ParametersState = {
  countOfWords: 2,
  inputText: null,
  submitNonce: 0,
};

export const parametersSlice = createSlice({
  name: "parameters",
  initialState,
  reducers: {
    setCount: (state, action: PayloadAction<number>) => {
      state.countOfWords = action.payload;
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

export const { setCount, setInputText, bumpSubmit, clearParameters } = parametersSlice.actions;
export default parametersSlice.reducer;
