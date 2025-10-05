import React from "react";

export default function AudienceSelect({ audience, setAudience }) {
  return (
    <select
      value={audience}
      onChange={(e) => setAudience(e.target.value)}
      className="bg-surface0 border border-surface2 rounded-xl p-3 text-text focus:outline-none focus:ring-2 focus:ring-mauve"
    >
      <option value="scientist">Scientist 🧬</option>
      <option value="manager">Manager 📊</option>
      <option value="architect">Mission Architect 🛰️</option>
    </select>
  );
}
