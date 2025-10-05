import React, { useState } from "react";

export default function SearchBar({ onSearch }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (value.trim()) onSearch(value);
  }

  return (
    <form onSubmit={handleSubmit} className="flex-1">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="🔍 Search NASA bioscience studies..."
        className="w-full bg-surface0 text-text border border-surface2 rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-blue"
      />
    </form>
  );
}
