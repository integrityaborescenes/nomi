import { configureStore } from "@reduxjs/toolkit";
import parametersSlice from "./slices/parametersSlice.ts";

export const store = configureStore({
  reducer: {
    parametersSlice: parametersSlice,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
