import { TULANE, UI } from './colors';

export default function MDPDiagram() {
  const W = 960, H = 520;

  const ballCx = 185, ballCy = 265, ballR = 100;
  const envX = 580, envY = 165, envW = 310, envH = 200;
  const envCx = envX + envW / 2;
  const midX = (ballCx + ballR + envX) / 2;

  // Arc geometry — symmetric cubic beziers
  // State arc: env top-left → pokeball top-right, curving up
  const sX1 = envX + 30, sY1 = envY;           // start
  const sX2 = ballCx + 60, sY2 = ballCy - ballR - 40; // end
  const sCpY = 40;                               // control point Y (how high)

  // Reward arc: env bottom-left → pokeball bottom-right, curving down
  const rX1 = envX + 30, rY1 = envY + envH;     // start
  const rX2 = ballCx + 60, rY2 = ballCy + ballR + 40; // end
  const rCpY = 490;                               // control point Y (how low)

  // Custom arrow triangle drawn manually at path endpoints
  function ArrowHead({ x, y, angle, color }: { x: number; y: number; angle: number; color: string }) {
    const size = 14;
    const rad = (angle * Math.PI) / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    // Triangle pointing in direction of angle
    const tip = { x, y };
    const left = { x: x - size * cos + (size * 0.45) * sin, y: y - size * sin - (size * 0.45) * cos };
    const right = { x: x - size * cos - (size * 0.45) * sin, y: y - size * sin + (size * 0.45) * cos };
    return <polygon points={`${tip.x},${tip.y} ${left.x},${left.y} ${right.x},${right.y}`} fill={color} />;
  }

  // Calculate tangent angle at the end of a cubic bezier
  // For C(x1,y1, cpx1,cpy1, cpx2,cpy2, x2,y2), tangent at t=1 is (x2-cpx2, y2-cpy2)
  const stateEndAngle = Math.atan2(sY2 - sCpY, sX2 - sX2) * (180 / Math.PI);
  // The state arc control points: (sX1, sCpY) and (sX2, sCpY), end at (sX2, sY2)
  // tangent at end = (sX2 - sX2, sY2 - sCpY) = (0, sY2 - sCpY) → pointing straight down
  // So angle = 90 degrees (pointing down)
  const stateArrowAngle = 90;

  // Reward arc: control points at rCpY, end at (rX2, rY2)
  // tangent at end = (0, rY2 - rCpY) → pointing up since rY2 < rCpY
  const rewardArrowAngle = -90;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
      <defs>
        <clipPath id="topHalf">
          <rect x={ballCx - ballR} y={ballCy - ballR} width={ballR * 2} height={ballR} />
        </clipPath>
        <clipPath id="bottomHalf">
          <rect x={ballCx - ballR} y={ballCy} width={ballR * 2} height={ballR} />
        </clipPath>
      </defs>

      {/* ═══ TITLE ═══ */}
      <text x={W / 2} y={30} textAnchor="middle" fontSize={28} fontWeight={700} fill={TULANE.green}>
        Markov Decision Process
      </text>

      {/* ═══ STATE ARC — Environment → Agent (top) ═══ */}
      <path
        d={`M ${sX1},${sY1} C ${sX1},${sCpY} ${sX2},${sCpY} ${sX2},${sY2}`}
        fill="none" stroke={TULANE.blue} strokeWidth={3.5}
      />
      <ArrowHead x={sX2} y={sY2} angle={stateArrowAngle} color={TULANE.blue} />

      {/* State label box — centered on the arc peak */}
      <rect x={midX - 65} y={sCpY - 6} width={130} height={34} rx={8}
        fill="white" stroke={TULANE.blue} strokeWidth={2} />
      <text x={midX} y={sCpY + 16} textAnchor="middle" fontSize={16} fontWeight={700} fill={TULANE.blue}>
        State  sₜ₊₁
      </text>
      <text x={midX} y={sCpY + 42} textAnchor="middle" fontSize={13} fontWeight={600} fill={UI.text}>
        Discretized battle state
      </text>

      {/* ═══ REWARD ARC — Environment → Agent (bottom) ═══ */}
      <path
        d={`M ${rX1},${rY1} C ${rX1},${rCpY} ${rX2},${rCpY} ${rX2},${rY2}`}
        fill="none" stroke="#E65100" strokeWidth={3.5}
      />
      <ArrowHead x={rX2} y={rY2} angle={rewardArrowAngle} color="#E65100" />

      {/* Reward label box — centered on the arc trough */}
      <rect x={midX - 75} y={rCpY - 28} width={150} height={34} rx={8}
        fill="white" stroke="#E65100" strokeWidth={2} />
      <text x={midX} y={rCpY - 6} textAnchor="middle" fontSize={16} fontWeight={700} fill="#E65100">
        Reward  rₜ₊₁
      </text>
      <text x={midX} y={rCpY + 20} textAnchor="middle" fontSize={13} fontWeight={600} fill={UI.text}>
        ±1 terminal + dense shaping
      </text>

      {/* ═══ ACTION ARROW — Agent → Environment (straight) ═══ */}
      <line x1={ballCx + ballR + 8} y1={ballCy}
            x2={envX - 18} y2={ballCy}
        stroke={TULANE.green} strokeWidth={3.5} />
      {/* Green arrow pointing right (0 degrees) */}
      <ArrowHead x={envX - 6} y={ballCy} angle={0} color={TULANE.green} />

      {/* Action label box — centered on the line */}
      <rect x={midX - 65} y={ballCy - 18} width={130} height={34} rx={8}
        fill="white" stroke={TULANE.green} strokeWidth={2} />
      <text x={midX} y={ballCy + 5} textAnchor="middle" fontSize={16} fontWeight={700} fill={TULANE.dark}>
        Action  aₜ
      </text>
      <text x={midX} y={ballCy + 30} textAnchor="middle" fontSize={13} fontWeight={600} fill={UI.text}>
        Move 1–4  |  Switch
      </text>

      {/* ═══ AGENT — POKEBALL ═══ */}
      <text x={ballCx} y={ballCy - ballR - 48} textAnchor="middle" fontSize={28} fontWeight={700} fill={UI.text}>
        Agent
      </text>

      <circle cx={ballCx} cy={ballCy} r={ballR} fill="#CC0000" clipPath="url(#topHalf)" />
      <circle cx={ballCx} cy={ballCy} r={ballR} fill="#FFFFFF" clipPath="url(#bottomHalf)" />
      <circle cx={ballCx} cy={ballCy} r={ballR} fill="none" stroke="#222" strokeWidth={3.5} />
      <line x1={ballCx - ballR} y1={ballCy} x2={ballCx + ballR} y2={ballCy}
        stroke="#222" strokeWidth={6} />
      <circle cx={ballCx} cy={ballCy} r={22} fill="#FFFFFF" stroke="#222" strokeWidth={4} />
      <circle cx={ballCx} cy={ballCy} r={10} fill="#FFFFFF" stroke="#222" strokeWidth={3} />

      <text x={ballCx} y={ballCy - 48} textAnchor="middle" fontSize={15} fontWeight={700} fill="white">
        Tabular Q-Learning
      </text>
      <text x={ballCx} y={ballCy - 30} textAnchor="middle" fontSize={13} fontWeight={600} fill="rgba(255,255,255,0.9)">
        Policy
      </text>
      <text x={ballCx} y={ballCy + 45} textAnchor="middle" fontSize={14} fontWeight={700} fill="#333">
        π(s) = argmax Q(s,a)
      </text>
      <text x={ballCx} y={ballCy + 66} textAnchor="middle" fontSize={12} fontWeight={600} fill={UI.light}>
        Watkins Q(λ)
      </text>

      {/* ═══ ENVIRONMENT BOX ═══ */}
      <text x={envCx} y={envY - 22} textAnchor="middle" fontSize={28} fontWeight={700} fill={UI.text}>
        Environment
      </text>

      <rect x={envX} y={envY} width={envW} height={envH} rx={16}
        fill={TULANE.blue} stroke="#2a6cb8" strokeWidth={2} />
      <text x={envCx} y={envY + 42} textAnchor="middle" fontSize={22} fontWeight={700} fill="white">
        Pokémon Showdown
      </text>
      <text x={envCx} y={envY + 68} textAnchor="middle" fontSize={20} fontWeight={700} fill="white">
        Server
      </text>
      <line x1={envX + 30} y1={envY + 84} x2={envX + envW - 30} y2={envY + 84}
        stroke="rgba(255,255,255,0.3)" strokeWidth={1.5} />
      <text x={envCx} y={envY + 112} textAnchor="middle" fontSize={16} fontWeight={600} fill="rgba(255,255,255,0.95)">
        Gen 4 OU Random Battles
      </text>
      <text x={envCx} y={envY + 150} textAnchor="middle" fontSize={15} fill="rgba(255,255,255,0.85)">
        P(s′ | s, a) — stochastic
      </text>
      <text x={envCx} y={envY + 175} textAnchor="middle" fontSize={13} fill="rgba(255,255,255,0.65)">
        partially observable transitions
      </text>
    </svg>
  );
}
