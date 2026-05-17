import Titlebar from "./components/Titlebar/Titlebar.tsx";
import PhraseInput from "./components/PhraseInput/PhraseInput.tsx";
import StyleSelector from "./components/StyleSelector/StyleSelector.tsx";
import ResultDisplay from "./components/ResultDisplay/ResultDisplay.tsx";

function App() {
  return (
    <>
      <Titlebar />
      <main>
        <PhraseInput />
        <StyleSelector />
        <ResultDisplay />
      </main>
    </>
  );
}

export default App;
