import { TULANE, POKEMON, MODELS, UI } from './colors';

export default function SmartInitDiagram() {
  const W = 920, H = 320;
  const tableW = 340, tableH = 220;
  const leftX = 40, rightX = 540, tableY = 70;
  const cols = 5, rows = 5;
  const cellW = tableW / cols, cellH = tableH / rows;

  const zeroVals = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];

  const smartVals = [
    [0.008, -0.003, 0.010, -0.007, 0.002],
    [-0.005, 0.009, 0.001, 0.006, -0.010],
    [0.007, -0.002, -0.008, 0.010, 0.004],
    [0.003, 0.010, -0.006, -0.001, 0.009],
    [-0.009, 0.005, 0.008, -0.004, 0.007],
  ];

  const actionLabels = ['Move 1', 'Move 2', 'Move 3', 'Move 4', 'SWITCH'];

  function valToColor(v: number, isZero: boolean): string {
    if (isZero) return '#ecf0f1';
    const t = (v + 0.01) / 0.02;
    if (t > 0.5) {
      const s = (t - 0.5) * 2;
      return `rgb(${Math.round(232 - s * 60)}, ${Math.round(245 - s * 20)}, ${Math.round(233 - s * 50)})`;
    }
    const s = (0.5 - t) * 2;
    return `rgb(${Math.round(255 - s * 10)}, ${Math.round(235 - s * 40)}, ${Math.round(238 - s * 50)})`;
  }

  function renderTable(
    x: number, vals: number[][], isZero: boolean, title: string, subtitle: string, titleColor: string,
  ) {
    return (
      <g>
        {/* Title */}
        <text x={x + tableW / 2} y={tableY - 48} textAnchor="middle" fontSize={20} fontWeight={700} fill={titleColor}>
          {title}
        </text>
        <text x={x + tableW / 2} y={tableY - 30} textAnchor="middle" fontSize={12} fill={UI.light}>
          {subtitle}
        </text>

        {/* Column headers */}
        {actionLabels.map((label, c) => (
          <text key={c} x={x + c * cellW + cellW / 2} y={tableY - 6} textAnchor="middle" fontSize={9}
            fontWeight={600} fill={c === 4 ? POKEMON.fighting : UI.text}
          >
            {label}
          </text>
        ))}

        {/* Row labels */}
        {['s₁', 's₂', 's₃', 's₄', 's₅'].map((label, r) => (
          <text key={r} x={x - 16} y={tableY + r * cellH + cellH / 2 + 4}
            textAnchor="middle" fontSize={12} fill={UI.light}
          >
            {label}
          </text>
        ))}

        {/* Table border */}
        <rect x={x} y={tableY} width={tableW} height={tableH} rx={4} fill="none" stroke={UI.border} strokeWidth={2} />

        {/* Cells */}
        {vals.map((row, r) =>
          row.map((v, c) => {
            const cx = x + c * cellW, cy = tableY + r * cellH;
            return (
              <g key={`${r}-${c}`}>
                <rect x={cx} y={cy} width={cellW} height={cellH}
                  fill={valToColor(v, isZero)} stroke={UI.border} strokeWidth={0.5}
                />
                <text x={cx + cellW / 2} y={cy + cellH / 2 + 4}
                  textAnchor="middle" fontSize={isZero ? 13 : 10}
                  fill={isZero ? '#bdc3c7' : (v > 0 ? TULANE.green : v < 0 ? POKEMON.fighting : UI.light)}
                  fontWeight={isZero ? 400 : 600}
                >
                  {isZero ? '0.00' : (v > 0 ? '+' : '') + v.toFixed(3)}
                </text>
              </g>
            );
          })
        )}

        {/* Highlight best action per row for smart init */}
        {!isZero && vals.map((row, r) => {
          const maxC = row.indexOf(Math.max(...row));
          return (
            <rect key={r}
              x={x + maxC * cellW + 1} y={tableY + r * cellH + 1}
              width={cellW - 2} height={cellH - 2}
              rx={3} fill="none" stroke={TULANE.kelly} strokeWidth={2.5}
            />
          );
        })}
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
      {/* Zero Init Table */}
      {renderTable(leftX + 20, zeroVals, true, 'Zero Initialization', 'Q(s, a) = 0 for all entries', MODELS.flat_zero)}

      {/* Smart Init Table */}
      {renderTable(rightX, smartVals, false, 'Heuristic Smart Init', 'Q₀(s,a) ∈ [−0.01, +0.01]', TULANE.green)}

      {/* VS divider */}
      <circle cx={W / 2} cy={tableY + tableH / 2} r={24} fill={POKEMON.fighting} />
      <text x={W / 2} y={tableY + tableH / 2 + 7} textAnchor="middle" fontSize={16} fontWeight={700} fill="white">
        vs
      </text>

    </svg>
  );
}
