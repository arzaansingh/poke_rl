import { TULANE, POKEMON, UI } from './colors';

export default function ArchitectureDiagram() {
  const W = 700, H = 630;

  const ENV_COLOR = TULANE.kelly;
  const ENV_BG = TULANE.lightGreen;
  const MASTER_COLOR = TULANE.blue;
  const MASTER_BG = TULANE.lightBlue;
  const SUB_COLOR = '#8E44AD';
  const SUB_BG = '#F3E5F5';
  const MOVE_COLOR = TULANE.green;
  const MOVE_BG = '#E8F5E9';
  const SWITCH_COLOR = POKEMON.fire;
  const SWITCH_BG = '#FFF3E0';
  const LEARN_COLOR = '#636e72';
  const LEARN_BG = '#f0f0f0';

  const cx = W / 2;
  const boxH = 70;
  const wideW = 320;
  const gap = 32; // minimum spacing (matches Env→Master)

  // Vertical positions — gap=32 between every element
  const envY = 18;
  // env bottom = 18+70 = 88
  const masterY = 88 + gap;              // 120
  const masterH = boxH + 10;             // 80, bottom = 200
  const diamondY = 200 + gap + 34;       // 266 (center), dS=34 so top=232, bottom=300
  const dS = 34;
  const branchY = 300 + gap;             // 332
  // branch bottom = 332+70 = 402
  const execSwitchY = 402 + gap;         // 434, h=60, bottom=494
  const learnY = 494 + gap + 10;         // 536

  // Horizontal branch spread — symmetric around center
  const branchOffset = 190;
  const leftX = cx - branchOffset;   // 160
  const rightX = cx + branchOffset;  // 540
  const branchW = 220;

  // Learning box must span from leftX to rightX
  const learnLeft = leftX - branchW / 2 + 10;  // 40
  const learnRight = rightX + branchW / 2 - 10; // 730
  const learnW = learnRight - learnLeft;         // 690

  function Arrow({ x1, y1, x2, y2, color = UI.text }: {
    x1: number; y1: number; x2: number; y2: number; color?: string;
  }) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    const ux = dx / len, uy = dy / len;
    const aSize = 10;
    const px = -uy * aSize, py = ux * aSize;
    const shaftEnd = { x: x2 - ux * 14, y: y2 - uy * 14 };
    return (
      <g>
        <line x1={x1} y1={y1} x2={shaftEnd.x} y2={shaftEnd.y}
          stroke={color} strokeWidth={3} />
        <polygon
          points={`${x2},${y2} ${x2 - ux * 14 + px},${y2 - uy * 14 + py} ${x2 - ux * 14 - px},${y2 - uy * 14 - py}`}
          fill={color} />
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', margin: '0 auto', maxWidth: W }}>

      {/* ═══ BATTLE ENVIRONMENT ═══ */}
      <rect x={cx - wideW / 2} y={envY} width={wideW} height={boxH} rx={14}
        fill={ENV_BG} stroke={ENV_COLOR} strokeWidth={2.5} />
      <text x={cx} y={envY + 28} textAnchor="middle" fontSize={20} fontWeight={700} fill={ENV_COLOR}>
        Battle Environment
      </text>
      <text x={cx} y={envY + 50} textAnchor="middle" fontSize={13} fill={UI.light}>
        Pokémon Showdown + poke-env
      </text>

      <Arrow x1={cx} y1={envY + boxH} x2={cx} y2={masterY} color={ENV_COLOR} />

      {/* ═══ MASTER Q-TABLE ═══ */}
      <rect x={cx - wideW / 2} y={masterY} width={wideW} height={masterH} rx={14}
        fill={MASTER_BG} stroke={MASTER_COLOR} strokeWidth={2.5} />
      <text x={cx} y={masterY + 26} textAnchor="middle" fontSize={20} fontWeight={700} fill={MASTER_COLOR}>
        Master Q-Table
      </text>
      <text x={cx} y={masterY + 46} textAnchor="middle" fontSize={13} fontWeight={600} fill={UI.text}>
        Q(s_master, a) · 20-tuple state
      </text>
      <text x={cx} y={masterY + 64} textAnchor="middle" fontSize={11} fill={UI.light}>
        Move 1 · Move 2 · Move 3 · Move 4 · Trigger-Switch
      </text>

      <Arrow x1={cx} y1={masterY + masterH} x2={cx} y2={diamondY - dS} color={MASTER_COLOR} />

      {/* ═══ DECISION DIAMOND ═══ */}
      <polygon
        points={`${cx},${diamondY - dS} ${cx + dS * 1.3},${diamondY} ${cx},${diamondY + dS} ${cx - dS * 1.3},${diamondY}`}
        fill="white" stroke={TULANE.dark} strokeWidth={2.5} />
      <text x={cx} y={diamondY + 6} textAnchor="middle" fontSize={15} fontWeight={700} fill={TULANE.dark}>
        Switch?
      </text>

      {/* ═══ LEFT BRANCH: NO → Execute Move ═══ */}
      <line x1={cx - dS * 1.3} y1={diamondY} x2={leftX} y2={diamondY}
        stroke={MOVE_COLOR} strokeWidth={3} />
      <text x={(cx - dS * 1.3 + leftX) / 2} y={diamondY - 12} textAnchor="middle" fontSize={15} fontWeight={700} fill={MOVE_COLOR}>
        No
      </text>
      <Arrow x1={leftX} y1={diamondY} x2={leftX} y2={branchY} color={MOVE_COLOR} />

      <rect x={leftX - branchW / 2} y={branchY} width={branchW} height={boxH} rx={14}
        fill={MOVE_BG} stroke={MOVE_COLOR} strokeWidth={2.5} />
      <text x={leftX} y={branchY + 28} textAnchor="middle" fontSize={19} fontWeight={700} fill={MOVE_COLOR}>
        Execute Move
      </text>
      <text x={leftX} y={branchY + 50} textAnchor="middle" fontSize={13} fill={UI.light}>
        Send chosen move to server
      </text>

      {/* ═══ RIGHT BRANCH: YES → Sub-Table → Execute Switch ═══ */}
      <line x1={cx + dS * 1.3} y1={diamondY} x2={rightX} y2={diamondY}
        stroke={SWITCH_COLOR} strokeWidth={3} />
      <text x={(cx + dS * 1.3 + rightX) / 2} y={diamondY - 12} textAnchor="middle" fontSize={15} fontWeight={700} fill={SWITCH_COLOR}>
        Yes / Forced
      </text>
      <Arrow x1={rightX} y1={diamondY} x2={rightX} y2={branchY} color={SWITCH_COLOR} />

      <rect x={rightX - branchW / 2} y={branchY} width={branchW} height={boxH} rx={14}
        fill={SUB_BG} stroke={SUB_COLOR} strokeWidth={2.5} />
      <text x={rightX} y={branchY + 24} textAnchor="middle" fontSize={19} fontWeight={700} fill={SUB_COLOR}>
        Switch Sub-Table
      </text>
      <text x={rightX} y={branchY + 44} textAnchor="middle" fontSize={12} fontWeight={600} fill={UI.text}>
        Q(s_sub, m) · 17-tuple state
      </text>
      <text x={rightX} y={branchY + 60} textAnchor="middle" fontSize={11} fill={UI.light}>
        Hazard-aware · Evaluates bench
      </text>

      {/* Sub → Execute Switch */}
      <Arrow x1={rightX} y1={branchY + boxH} x2={rightX} y2={execSwitchY} color={SUB_COLOR} />

      <rect x={rightX - branchW / 2 + 10} y={execSwitchY} width={branchW - 20} height={60} rx={14}
        fill={SWITCH_BG} stroke={SWITCH_COLOR} strokeWidth={2.5} />
      <text x={rightX} y={execSwitchY + 26} textAnchor="middle" fontSize={17} fontWeight={700} fill={SWITCH_COLOR}>
        Execute Switch
      </text>
      <text x={rightX} y={execSwitchY + 46} textAnchor="middle" fontSize={12} fill={UI.light}>
        Best bench Pokémon
      </text>

      {/* ═══ BOTH → LEARNING UPDATE ═══ */}
      <Arrow x1={leftX} y1={branchY + boxH} x2={leftX} y2={learnY} color={LEARN_COLOR} />
      <Arrow x1={rightX} y1={execSwitchY + 60} x2={rightX} y2={learnY} color={LEARN_COLOR} />

      {/* Wide learning box spanning both branches */}
      <rect x={learnLeft} y={learnY} width={learnW} height={boxH} rx={14}
        fill={LEARN_BG} stroke={LEARN_COLOR} strokeWidth={2.5} />
      <text x={cx} y={learnY + 26} textAnchor="middle" fontSize={19} fontWeight={700} fill={UI.text}>
        Learning Update
      </text>
      <text x={cx} y={learnY + 46} textAnchor="middle" fontSize={13} fill={UI.light}>
        Reward + TD Error → Watkins Q(λ) Traces
      </text>
      <text x={cx} y={learnY + 62} textAnchor="middle" fontSize={11} fill={UI.light}>
        Update both Q-tables · Dense shaping + terminal ±1
      </text>

      {/* ═══ LOOP BACK: Next Turn ═══ */}
      <line x1={learnLeft} y1={learnY + boxH / 2}
        x2={24} y2={learnY + boxH / 2}
        stroke={ENV_COLOR} strokeWidth={2.5} strokeDasharray="8,5" />
      <line x1={24} y1={learnY + boxH / 2}
        x2={24} y2={envY + boxH / 2}
        stroke={ENV_COLOR} strokeWidth={2.5} strokeDasharray="8,5" />
      <line x1={24} y1={envY + boxH / 2}
        x2={cx - wideW / 2} y2={envY + boxH / 2}
        stroke={ENV_COLOR} strokeWidth={2.5} strokeDasharray="8,5" />
      <polygon points={`${cx - wideW / 2},${envY + boxH / 2} ${cx - wideW / 2 - 10},${envY + boxH / 2 - 6} ${cx - wideW / 2 - 10},${envY + boxH / 2 + 6}`}
        fill={ENV_COLOR} />
      <text x={10} y={(envY + learnY) / 2 + 30} textAnchor="middle" fontSize={15} fontWeight={700} fill={ENV_COLOR}
        transform={`rotate(-90, 10, ${(envY + learnY) / 2 + 30})`}>
        Next Turn
      </text>

      {/* ═══ LEGEND (top right) ═══ */}
      <g transform={`translate(${W - 155}, ${envY})`}>
        <rect x={-10} y={-8} width={160} height={170} rx={10} fill="#fafafa" stroke={UI.border} strokeWidth={1.5} />
        {[
          { color: ENV_COLOR, bg: ENV_BG, label: 'Environment' },
          { color: MASTER_COLOR, bg: MASTER_BG, label: 'Master' },
          { color: SUB_COLOR, bg: SUB_BG, label: 'Sub Layer' },
          { color: MOVE_COLOR, bg: MOVE_BG, label: 'Move Path' },
          { color: SWITCH_COLOR, bg: SWITCH_BG, label: 'Switch Path' },
          { color: LEARN_COLOR, bg: LEARN_BG, label: 'Learning' },
        ].map((item, i) => (
          <g key={i} transform={`translate(0, ${i * 26})`}>
            <rect width={16} height={16} rx={4} fill={item.bg} stroke={item.color} strokeWidth={2} />
            <text x={22} y={13} fontSize={15} fontWeight={700} fill={UI.text}>{item.label}</text>
          </g>
        ))}
      </g>

    </svg>
  );
}
