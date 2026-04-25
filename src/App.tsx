import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import Titlebar from "./components/Titlebar/Titlebar.tsx";
import PhraseInput from "./components/PhraseInput/PhraseInput.tsx";
import WordCountSelector from "./components/WordCountSelector/WordCountSelector.tsx";
import ResultDisplay from "./components/ResultDisplay/ResultDisplay.tsx";
import SetupWizard from "./components/SetupWizard/SetupWizard.tsx";

type SetupStatus = {
  ollamaInstalled: boolean;
  ollamaRunning: boolean;
  modelReady: boolean;
};

function App() {
  const [ready, setReady] = useState<boolean | null>(null);

  useEffect(() => {
    (async () => {
      const status = await invoke<SetupStatus>("check_setup_status");
      setReady(status.ollamaInstalled && status.ollamaRunning && status.modelReady);
    })();
  }, []);

  return (
    <>
      <Titlebar />
      <main>
        {ready === null ? null : ready ? (
          <>
            <PhraseInput />
            <WordCountSelector />
            <ResultDisplay />
          </>
        ) : (
          <SetupWizard onReady={() => setReady(true)} />
        )}
      </main>
    </>
  );
}

export default App;
