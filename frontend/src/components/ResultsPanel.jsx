// src/components/ResultsPanel.jsx
import React from "react";

export default function ResultsPanel({ summary, studies }) {
  const safeSummary = summary?.trim() || "";
  const safeStudies = Array.isArray(studies) ? studies : [];

  return (
    <div className="w-full max-w-3xl p-6 rounded-2xl bg-[#313244] shadow-lg border border-[#45475a]">
      {/* Summary */}
      <h2 className="text-xl font-semibold text-[#f5c2e7] mb-3">Summary</h2>
      <p className="text-[#cdd6f4] leading-relaxed">
        {safeSummary
          ? safeSummary.charAt(0).toUpperCase() + safeSummary.slice(1)
          : "🔍 No summary available. Try searching for something!"}
      </p>

      {/* Studies */}
      <h3 className="text-lg font-semibold text-[#94e2d5] mt-6 mb-2">
        Relevant Studies
      </h3>
      {safeStudies.length > 0 ? (
        <ul className="list-disc pl-6 space-y-1">
          {safeStudies.map((s, i) => (
            <li key={i}>
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#89b4fa] hover:underline"
              >
                {s.title}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[#a6adc8]">No studies retrieved yet.</p>
      )}
    </div>
  );
}
