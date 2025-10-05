// src/App.jsx
import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import AudienceSelect from "./components/AudienceSelect";
import ResultsPanel from "./components/ResultsPanel";

function App() {
  const [audience, setAudience] = useState("scientist");
  const [summary, setSummary] = useState("");   // default safe values
  const [studies, setStudies] = useState([]);

  return (
    <div className="min-h-screen bg-[#1e1e2e] text-[#cdd6f4] flex flex-col">
      {/* Header */}
      <header className="p-6 bg-[#181825] shadow-md text-center">
        <h1 className="text-3xl font-bold text-[#f5c2e7]">
          🚀 NASA SpaceApps AI Summarizer
        </h1>
        <p className="text-[#a6adc8]">Search • Summarize • Visualize</p>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 flex flex-col gap-6 items-center">
        <SearchBar
          onSearch={(query) => {
            // 🔹 placeholder demo handler — replace with API call later
            setSummary(`This is a demo summary for "${query}" tailored to ${audience}.`);
            setStudies([
              { title: "Study 1: Climate Impact", url: "https://example.com/study1" },
              { title: "Study 2: Space Bioscience", url: "https://example.com/study2" },
            ]);
          }}
        />

        <AudienceSelect value={audience} onChange={setAudience} />

        <ResultsPanel summary={summary} studies={studies} />
      </main>

      {/* Footer */}
      <footer className="p-4 bg-[#181825] text-center text-[#a6adc8]">
        Built with 💜 using React, TailwindCSS & FastAPI
      </footer>
    </div>
  );
}

export default App;
