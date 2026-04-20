import { useState, useRef, useCallback } from 'react';
import MDPDiagram from './components/MDPDiagram';
import SmartInitDiagram from './components/SmartInitDiagram';
import ArchitectureDiagram from './components/ArchitectureDiagram';

const tabs = [
  { id: 'mdp', label: 'MDP Diagram', Component: MDPDiagram },
  { id: 'smart-init', label: 'Smart Initialization', Component: SmartInitDiagram },
  { id: 'architecture', label: 'Hierarchical Architecture', Component: ArchitectureDiagram },
] as const;

function downloadSVG(svgEl: SVGSVGElement | null, filename: string) {
  if (!svgEl) return;
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], {
    type: 'image/svg+xml;charset=utf-8',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.svg`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadPNG(svgEl: SVGSVGElement | null, filename: string, scale = 4) {
  if (!svgEl) return;
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const svgData = new XMLSerializer().serializeToString(clone);
  const vb = svgEl.viewBox.baseVal;
  const canvas = document.createElement('canvas');
  canvas.width = vb.width * scale;
  canvas.height = vb.height * scale;
  const ctx = canvas.getContext('2d')!;
  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `${filename}.png`;
    a.click();
  };
  img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
}

export default function App() {
  const [active, setActive] = useState(tabs[0].id);
  const svgRef = useRef<SVGSVGElement>(null);

  const currentTab = tabs.find((t) => t.id === active)!;

  const wrapRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (node) {
        const svg = node.querySelector('svg');
        (svgRef as React.MutableRefObject<SVGSVGElement | null>).current = svg;
      }
    },
    [active],
  );

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <h1 style={{ fontSize: 28, color: '#006747', marginBottom: 8 }}>
        PokeAgent — Poster Visualizations
      </h1>
      <p style={{ color: '#666', marginBottom: 24 }}>
        Click a tab, then download as SVG or PNG (4x) for the poster.
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            style={{
              padding: '10px 20px',
              fontSize: 15,
              fontWeight: active === t.id ? 700 : 400,
              border: `2px solid ${active === t.id ? '#006747' : '#ccc'}`,
              borderRadius: 8,
              background: active === t.id ? '#006747' : 'white',
              color: active === t.id ? 'white' : '#333',
              cursor: 'pointer',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Download buttons */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <button
          onClick={() => downloadSVG(svgRef.current, currentTab.id)}
          style={{
            padding: '8px 18px', fontSize: 14, fontWeight: 600,
            border: '2px solid #418FDE', borderRadius: 8,
            background: '#418FDE', color: 'white', cursor: 'pointer',
          }}
        >
          Download SVG
        </button>
        <button
          onClick={() => downloadPNG(svgRef.current, currentTab.id)}
          style={{
            padding: '8px 18px', fontSize: 14, fontWeight: 600,
            border: '2px solid #43B02A', borderRadius: 8,
            background: '#43B02A', color: 'white', cursor: 'pointer',
          }}
        >
          Download PNG (4x)
        </button>
      </div>

      {/* SVG canvas */}
      <div
        ref={wrapRef}
        style={{
          border: '1px solid #e0e0e0',
          borderRadius: 12,
          padding: 16,
          background: 'white',
        }}
      >
        <currentTab.Component />
      </div>
    </div>
  );
}
