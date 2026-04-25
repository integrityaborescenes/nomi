import Titlebar from "./components/Titlebar/Titlebar.tsx";
import PhraseInput from "./components/PhraseInput/PhraseInput.tsx";
import WordCountSelector from "./components/WordCountSelector/WordCountSelector.tsx";
import ResultDisplay from "./components/ResultDisplay/ResultDisplay.tsx";

function App() {
  return (
    <>
      <Titlebar />
      <main>
        <PhraseInput />
        <WordCountSelector />
        <ResultDisplay />
      </main>
    </>
  );
}

export default App;
