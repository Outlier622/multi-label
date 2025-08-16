// App.jsx
import React, { useState } from "react";

export default function App() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [topk, setTopk] = useState(5);

  const onSelect = (e) => {
    setFiles([...e.target.files]);
    setResults([]);
    setError("");
  };

  const onPredict = async () => {
    if (!files.length) return;
    setLoading(true);
    setError("");
    setResults([]);
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("topk", String(topk));
    try {
      const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: form,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setResults(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 900,
        margin: "40px auto",
        fontFamily: "system-ui, -apple-system, Segoe UI, Roboto",
      }}
    >
      <h1>ViT 多标签推理 Demo</h1>
      <p>选择图片并点击推理。后端默认读取 per-class 阈值（若提供）。</p>

      <div style={{ margin: "12px 0" }}>
        <input type="file" multiple accept="image/*" onChange={onSelect} />
        <input
          type="number"
          min={1}
          max={10}
          value={topk}
          style={{ marginLeft: 8, width: 64 }}
          onChange={(e) => setTopk(Number(e.target.value))}
        />
        <button
          onClick={onPredict}
          disabled={!files.length || loading}
          style={{ marginLeft: 8 }}
        >
          {loading ? "推理中..." : "开始推理"}
        </button>
      </div>

      {error && <div style={{ color: "crimson" }}>错误：{error}</div>}

      {!!results.length && (
        <div style={{ marginTop: 16 }}>
          {results.map((r, idx) => (
            <div
              key={idx}
              style={{
                border: "1px solid #ddd",
                borderRadius: 12,
                padding: 12,
                marginBottom: 12,
              }}
            >
              <div style={{ fontWeight: 600 }}>{r.filename}</div>
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  flexWrap: "wrap",
                  marginTop: 8,
                }}
              >
                {r.topk.map((t, i) => (
                  <span
                    key={i}
                    style={{
                      background: "#f2f2f2",
                      borderRadius: 999,
                      padding: "4px 10px",
                    }}
                  >
                    {t.label}: {(t.prob * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
