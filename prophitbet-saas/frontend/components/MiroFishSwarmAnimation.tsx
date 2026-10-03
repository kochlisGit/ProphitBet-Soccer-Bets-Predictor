"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";

export interface SwarmMatchContext {
  homeTeam: string;
  awayTeam: string;
  league: string;
  probH: number;
  probD: number;
  probA: number;
  consensusLevel?: string;
}

interface Agent {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  ax: number;
  ay: number;
  persona: "tactical" | "sharp" | "morale" | "dynamics" | "synthesizer";
  color: string;
  radius: number;
  maxSpeed: number;
  maxForce: number;
  targetAttractor: "H" | "D" | "A";
  thought: string;
  trail: { x: number; y: number }[];
}

interface Attractor {
  key: "H" | "D" | "A";
  label: string;
  teamLabel: string;
  x: number;
  y: number;
  color: string;
  glowColor: string;
  targetProb: number;
  currentProb: number;
  radius: number;
}

const PERSONA_CONFIGS = {
  tactical: {
    name: "Tactical Strategist",
    color: "#10B981", // Emerald
    thoughts: [
      "Overloading left half-space to bypass low block",
      "Counter-press transition phase vulnerable on turnaround",
      "Central midfield 3v2 numerical advantage detected",
      "High defensive line caught out on vertical through-balls",
    ],
  },
  sharp: {
    name: "Quantitative Sharp",
    color: "#06B6D4", // Cyan
    thoughts: [
      "Market line implies 44% Home but Poisson predicts 51%",
      "Positive EV edge identified on the Asian handicap",
      "Public bias overvaluing brand name against regression",
      "Bookmaker margin discrepancy creating arbitrage window",
    ],
  },
  morale: {
    name: "Squad Morale Insider",
    color: "#A855F7", // Purple
    thoughts: [
      "Locker-room momentum riding high after comeback win",
      "Third match in 7 days; late-game fatigue risk elevated",
      "Captain returning from suspension boosts defensive communication",
      "Relegation fight urgency driving high-intensity duels",
    ],
  },
  dynamics: {
    name: "Match Dynamics Specialist",
    color: "#F43F5E", // Rose
    thoughts: [
      "Strict referee averaging 5.2 yellows per 90; tactical fouls penalised",
      "Waterlogged pitch slowing ball circulation and favoring aerial set-pieces",
      "Late-game volatility spikes if deadlock remains past 70'",
      "Early goal script drastically accelerates open-field transitions",
    ],
  },
  synthesizer: {
    name: "Consensus Synthesizer",
    color: "#F59E0B", // Gold
    thoughts: [
      "Weighing tactical dominance against market odds variance",
      "Bayesian fusion indicates high-conviction modal cluster",
      "Resolving tension between quantitative and physical intangibles",
      "Stabilizing emergent consensus trajectory across the swarm",
    ],
  },
};

export default function MiroFishSwarmAnimation({
  match,
  onProbabilitiesChange,
}: {
  match?: SwarmMatchContext;
  onProbabilitiesChange?: (probs: { H: number; D: number; A: number }) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Match State
  const [currentMatch, setCurrentMatch] = useState<SwarmMatchContext>(
    match || {
      homeTeam: "Manchester City",
      awayTeam: "Aston Villa",
      league: "Premier League",
      probH: 0.52,
      probD: 0.27,
      probA: 0.21,
      consensusLevel: "STRONG_CONSENSUS",
    }
  );

  // Simulation controls
  const [particleCount, setParticleCount] = useState<number>(140);
  const [simulationSpeed, setSimulationSpeed] = useState<number>(1.0);
  const [cohesionStrength, setCohesionStrength] = useState<number>(0.8);
  const [activeScenario, setActiveScenario] = useState<string>("equilibrium");
  const [hoveredAgent, setHoveredAgent] = useState<Agent | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [emergentProbs, setEmergentProbs] = useState({ H: 52, D: 27, A: 21 });
  const [fps, setFps] = useState(60);

  // Shockwaves
  const shockwavesRef = useRef<{ x: number; y: number; radius: number; maxRadius: number; opacity: number }[]>([]);

  // Update match when prop changes
  useEffect(() => {
    if (match) {
      setCurrentMatch(match);
      setActiveScenario("equilibrium");
    }
  }, [match]);

  // Main Canvas Animation Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsLastSample = performance.now();

    // Resize canvas with pixel ratio
    const resizeCanvas = () => {
      if (!canvas || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Initial Attractors layout
    const getAttractors = (): Attractor[] => {
      const w = canvas.width / (window.devicePixelRatio || 1);
      const h = canvas.height / (window.devicePixelRatio || 1);

      return [
        {
          key: "H",
          label: "HOME WIN (1)",
          teamLabel: currentMatch.homeTeam,
          x: w * 0.22,
          y: h * 0.50,
          color: "#10B981",
          glowColor: "rgba(16, 185, 129, 0.4)",
          targetProb: currentMatch.probH,
          currentProb: currentMatch.probH,
          radius: 36,
        },
        {
          key: "D",
          label: "DRAW (X)",
          teamLabel: "Stalemate / Neutral",
          x: w * 0.50,
          y: h * 0.28,
          color: "#F59E0B",
          glowColor: "rgba(245, 158, 11, 0.4)",
          targetProb: currentMatch.probD,
          currentProb: currentMatch.probD,
          radius: 32,
        },
        {
          key: "A",
          label: "AWAY WIN (2)",
          teamLabel: currentMatch.awayTeam,
          x: w * 0.78,
          y: h * 0.50,
          color: "#F43F5E",
          glowColor: "rgba(244, 63, 94, 0.4)",
          targetProb: currentMatch.probA,
          currentProb: currentMatch.probA,
          radius: 36,
        },
      ];
    };

    let attractors = getAttractors();

    // Generate initial agents
    const personasList: Agent["persona"][] = [
      "tactical",
      "tactical",
      "sharp",
      "sharp",
      "morale",
      "dynamics",
      "synthesizer",
    ];

    const pickAttractor = (probs: { H: number; D: number; A: number }): "H" | "D" | "A" => {
      const r = Math.random();
      if (r < probs.H) return "H";
      if (r < probs.H + probs.D) return "D";
      return "A";
    };

    const agents: Agent[] = [];
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);

    for (let i = 0; i < particleCount; i++) {
      const persona = personasList[i % personasList.length];
      const targetAttractor = pickAttractor({
        H: currentMatch.probH,
        D: currentMatch.probD,
        A: currentMatch.probA,
      });

      const config = PERSONA_CONFIGS[persona];
      const thought = config.thoughts[Math.floor(Math.random() * config.thoughts.length)];

      agents.push({
        id: i + 1,
        x: w * 0.5 + (Math.random() - 0.5) * 200,
        y: h * 0.5 + (Math.random() - 0.5) * 200,
        vx: (Math.random() - 0.5) * 3,
        vy: (Math.random() - 0.5) * 3,
        ax: 0,
        ay: 0,
        persona,
        color: config.color,
        radius: persona === "synthesizer" ? 4.5 : 3.2,
        maxSpeed: (2.4 + Math.random() * 1.2) * simulationSpeed,
        maxForce: 0.12,
        targetAttractor,
        thought,
        trail: [],
      });
    }

    // Mouse tracking
    let mouse = { x: -1000, y: -1000, isDown: false };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;

      // Check hover on agents
      let found: Agent | null = null;
      for (const a of agents) {
        const d = Math.hypot(a.x - mouse.x, a.y - mouse.y);
        if (d < 15) {
          found = a;
          break;
        }
      }
      setHoveredAgent(found);
    };

    const handleMouseDown = (e: MouseEvent) => {
      mouse.isDown = true;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Trigger Shockwave
      shockwavesRef.current.push({
        x,
        y,
        radius: 5,
        maxRadius: 180,
        opacity: 0.9,
      });

      // Select agent if clicked
      for (const a of agents) {
        const d = Math.hypot(a.x - x, a.y - y);
        if (d < 20) {
          setSelectedAgent(a);
          return;
        }
      }
    };

    const handleMouseUp = () => {
      mouse.isDown = false;
    };

    const handleMouseLeave = () => {
      mouse.x = -1000;
      mouse.y = -1000;
      setHoveredAgent(null);
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    canvas.addEventListener("mouseleave", handleMouseLeave);

    // Render loop
    const render = (now: number) => {
      frameCount++;
      if (now - fpsLastSample >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsLastSample)));
        frameCount = 0;
        fpsLastSample = now;
      }

      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      const cw = canvas.width / (window.devicePixelRatio || 1);
      const ch = canvas.height / (window.devicePixelRatio || 1);

      // Re-center attractors if size changed
      attractors[0].x = cw * 0.22;
      attractors[0].y = ch * 0.50;
      attractors[1].x = cw * 0.50;
      attractors[1].y = ch * 0.28;
      attractors[2].x = cw * 0.78;
      attractors[2].y = ch * 0.50;

      // Clear with trail fade
      ctx.fillStyle = "rgba(9, 9, 11, 0.35)";
      ctx.fillRect(0, 0, cw, ch);

      // Draw faint soccer pitch tactical grid
      drawTacticalPitch(ctx, cw, ch);

      // Draw Attractors (Gravitational Outcome Hubs)
      for (const att of attractors) {
        drawAttractor(ctx, att, now);
      }

      // Update shockwaves
      for (let i = shockwavesRef.current.length - 1; i >= 0; i--) {
        const sw = shockwavesRef.current[i];
        sw.radius += 5.5 * simulationSpeed;
        sw.opacity *= 0.94;

        ctx.save();
        ctx.beginPath();
        ctx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(168, 85, 247, ${sw.opacity})`;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.restore();

        // Repel agents within shockwave
        for (const a of agents) {
          const d = Math.hypot(a.x - sw.x, a.y - sw.y);
          if (Math.abs(d - sw.radius) < 25) {
            const angle = Math.atan2(a.y - sw.y, a.x - sw.x);
            a.vx += Math.cos(angle) * 3.5;
            a.vy += Math.sin(angle) * 3.5;
          }
        }

        if (sw.opacity < 0.04 || sw.radius > sw.maxRadius) {
          shockwavesRef.current.splice(i, 1);
        }
      }

      // Draw Synaptic Debate Links between nearby agents
      drawSynapticLinks(ctx, agents);

      // Counts near each attractor to determine live emergent probability
      let counts = { H: 0, D: 0, A: 0 };

      // Physics update for agents
      for (let i = 0; i < agents.length; i++) {
        const a = agents[i];

        // 1. Gravitational Attraction to Target Outcome Hub
        const targetAtt = attractors.find((att) => att.key === a.targetAttractor) || attractors[0];
        const dx = targetAtt.x - a.x;
        const dy = targetAtt.y - a.y;
        const distToAtt = Math.hypot(dx, dy);

        // Steering force towards attractor
        if (distToAtt > 10) {
          const speed = distToAtt < 80 ? (a.maxSpeed * distToAtt) / 80 : a.maxSpeed;
          const desiredX = (dx / distToAtt) * speed;
          const desiredY = (dy / distToAtt) * speed;
          const steerX = desiredX - a.vx;
          const steerY = desiredY - a.vy;
          a.ax += steerX * 0.04 * cohesionStrength;
          a.ay += steerY * 0.04 * cohesionStrength;
        }

        // 2. Flocking Boids behaviors: Separation, Alignment, Cohesion
        let sepX = 0,
          sepY = 0,
          sepCount = 0;
        let alignX = 0,
          alignY = 0,
          alignCount = 0;
        let cohX = 0,
          cohY = 0,
          cohCount = 0;

        for (let j = 0; j < agents.length; j++) {
          if (i === j) continue;
          const b = agents[j];
          const d = Math.hypot(a.x - b.x, a.y - b.y);

          // Separation (avoid crowding)
          if (d > 0 && d < 24) {
            sepX += (a.x - b.x) / d;
            sepY += (a.y - b.y) / d;
            sepCount++;
          }

          // Neighbor flocking
          if (d > 0 && d < 60) {
            alignX += b.vx;
            alignY += b.vy;
            alignCount++;

            cohX += b.x;
            cohY += b.y;
            cohCount++;
          }
        }

        if (sepCount > 0) {
          a.ax += (sepX / sepCount) * 0.15;
          a.ay += (sepY / sepCount) * 0.15;
        }

        if (alignCount > 0) {
          const avgVx = (alignX / alignCount) * 0.05;
          const avgVy = (alignY / alignCount) * 0.05;
          a.ax += (avgVx - a.vx) * 0.03;
          a.ay += (avgVy - a.vy) * 0.03;
        }

        if (cohCount > 0) {
          const centerDistX = cohX / cohCount - a.x;
          const centerDistY = cohY / cohCount - a.y;
          a.ax += centerDistX * 0.001 * cohesionStrength;
          a.ay += centerDistY * 0.001 * cohesionStrength;
        }

        // 3. Mouse Interaction (Pointer repulsion / vortex)
        const mouseDist = Math.hypot(a.x - mouse.x, a.y - mouse.y);
        if (mouseDist < 90) {
          const force = (1 - mouseDist / 90) * 4.0;
          const angle = Math.atan2(a.y - mouse.y, a.x - mouse.x);
          a.ax += Math.cos(angle) * force;
          a.ay += Math.sin(angle) * force;
        }

        // 4. Brownian motion / debate agitation
        a.ax += (Math.random() - 0.5) * 0.35;
        a.ay += (Math.random() - 0.5) * 0.35;

        // Apply velocities
        a.vx += a.ax * simulationSpeed;
        a.vy += a.ay * simulationSpeed;

        // Speed limit
        const curSpeed = Math.hypot(a.vx, a.vy);
        const maxS = a.maxSpeed * simulationSpeed;
        if (curSpeed > maxS) {
          a.vx = (a.vx / curSpeed) * maxS;
          a.vy = (a.vy / curSpeed) * maxS;
        }

        a.x += a.vx;
        a.y += a.vy;

        // Reset acceleration
        a.ax = 0;
        a.ay = 0;

        // Bounce off canvas boundaries
        const margin = 20;
        if (a.x < margin) {
          a.x = margin;
          a.vx *= -0.7;
        }
        if (a.x > cw - margin) {
          a.x = cw - margin;
          a.vx *= -0.7;
        }
        if (a.y < margin) {
          a.y = margin;
          a.vy *= -0.7;
        }
        if (a.y > ch - margin) {
          a.y = ch - margin;
          a.vy *= -0.7;
        }

        // Trail history
        if (frameCount % 2 === 0) {
          a.trail.push({ x: a.x, y: a.y });
          if (a.trail.length > 5) a.trail.shift();
        }

        // Measure nearest attractor for emergent probabilities
        let closestDist = Infinity;
        let closestKey: "H" | "D" | "A" = "H";
        for (const att of attractors) {
          const d = Math.hypot(a.x - att.x, a.y - att.y);
          if (d < closestDist) {
            closestDist = d;
            closestKey = att.key;
          }
        }
        counts[closestKey]++;

        // Draw Agent Particle
        drawAgent(ctx, a, a === selectedAgent || a === hoveredAgent);
      }

      // Calculate spatial emergent probabilities
      const totalAgents = agents.length || 1;
      const curH = Math.round((counts.H / totalAgents) * 100);
      const curD = Math.round((counts.D / totalAgents) * 100);
      const curA = Math.max(0, 100 - curH - curD);

      setEmergentProbs({ H: curH, D: curD, A: curA });

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", resizeCanvas);
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [currentMatch, particleCount, simulationSpeed, cohesionStrength, selectedAgent]);

  // Scenario Triggers
  const applyScenario = (scenarioId: string) => {
    setActiveScenario(scenarioId);
    let newH = currentMatch.probH;
    let newD = currentMatch.probD;
    let newA = currentMatch.probA;

    if (scenarioId === "home_goal") {
      newH = 0.76;
      newD = 0.16;
      newA = 0.08;
    } else if (scenarioId === "red_card_home") {
      newH = 0.18;
      newD = 0.34;
      newA = 0.48;
    } else if (scenarioId === "muddy_pitch") {
      newH = 0.32;
      newD = 0.48;
      newA = 0.20;
    } else if (scenarioId === "sharp_steam") {
      newH = 0.58;
      newD = 0.24;
      newA = 0.18;
    } else {
      // equilibrium
      newH = match?.probH || 0.52;
      newD = match?.probD || 0.27;
      newA = match?.probA || 0.21;
    }

    setCurrentMatch((prev) => ({
      ...prev,
      probH: newH,
      probD: newD,
      probA: newA,
    }));

    if (onProbabilitiesChange) {
      onProbabilitiesChange({ H: newH, D: newD, A: newA });
    }
  };

  return (
    <div className="space-y-4">
      {/* Visual Arena Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-zinc-900/90 p-4 rounded-xl border border-zinc-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-purple-500 animate-ping" />
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>🌌</span> MiroFish Swarm Neural Arena
            </h3>
            <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-purple-500/10 text-purple-300 border border-purple-500/30 rounded-md">
              LIVE 60FPS
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-0.5">
            {currentMatch.homeTeam} vs {currentMatch.awayTeam} · True flocking physics with Bayesian gravitational outcome attractors.
          </p>
        </div>

        {/* Live Emergent Probability Bar */}
        <div className="flex items-center gap-4 bg-zinc-950/80 px-4 py-2 rounded-lg border border-zinc-800/80">
          <div className="text-center font-mono">
            <span className="text-[10px] text-zinc-500 block uppercase">Home Win</span>
            <span className="text-sm font-bold text-emerald-400">{emergentProbs.H}%</span>
          </div>
          <span className="text-zinc-700">|</span>
          <div className="text-center font-mono">
            <span className="text-[10px] text-zinc-500 block uppercase">Draw</span>
            <span className="text-sm font-bold text-amber-400">{emergentProbs.D}%</span>
          </div>
          <span className="text-zinc-700">|</span>
          <div className="text-center font-mono">
            <span className="text-[10px] text-zinc-500 block uppercase">Away Win</span>
            <span className="text-sm font-bold text-rose-400">{emergentProbs.A}%</span>
          </div>
        </div>
      </div>

      {/* Main Canvas Simulation Arena */}
      <div
        ref={containerRef}
        className="relative w-full h-[520px] rounded-2xl overflow-hidden bg-zinc-950 border border-zinc-800 shadow-2xl group"
      >
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />

        {/* Top-left Arena Telemetry Overlay */}
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 pointer-events-none">
          <div className="backdrop-blur-md bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-[11px] font-mono text-zinc-300 flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              {particleCount} Agents
            </span>
            <span className="text-zinc-600">·</span>
            <span>{fps} FPS</span>
            <span className="text-zinc-600">·</span>
            <span className="text-purple-300 capitalize">{activeScenario.replace("_", " ")}</span>
          </div>
        </div>

        {/* Top-right Persona Legend */}
        <div className="absolute top-4 right-4 z-10 hidden md:flex items-center gap-2 backdrop-blur-md bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-lg text-[10px] font-medium text-zinc-300">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400" /> Tactical
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-cyan-400" /> Sharp
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-400" /> Morale
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-400" /> Dynamics
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-400" /> Synthesizer
          </span>
        </div>

        {/* Bottom Interactive Inspector HUD (Hovered or Selected Agent) */}
        {(selectedAgent || hoveredAgent) && (
          <div className="absolute bottom-4 left-4 right-4 z-10 backdrop-blur-md bg-zinc-900/90 border border-purple-500/40 p-3.5 rounded-xl text-xs text-white shadow-xl flex items-center justify-between gap-4 transition-all">
            <div className="flex items-center gap-3">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm"
                style={{ backgroundColor: `${(selectedAgent || hoveredAgent)?.color}20`, color: (selectedAgent || hoveredAgent)?.color, border: `1px solid ${(selectedAgent || hoveredAgent)?.color}40` }}
              >
                #{(selectedAgent || hoveredAgent)?.id}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-white">
                    {PERSONA_CONFIGS[(selectedAgent || hoveredAgent)!.persona].name}
                  </span>
                  <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-zinc-800 text-zinc-300">
                    Target: {(selectedAgent || hoveredAgent)?.targetAttractor === "H" ? "Home Win" : (selectedAgent || hoveredAgent)?.targetAttractor === "A" ? "Away Win" : "Draw"}
                  </span>
                </div>
                <p className="text-zinc-300 text-xs italic mt-0.5">
                  "{(selectedAgent || hoveredAgent)?.thought}"
                </p>
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-3 text-[11px] font-mono text-zinc-400 border-l border-zinc-800 pl-4">
              <span>Velocity: {Math.hypot((selectedAgent || hoveredAgent)!.vx, (selectedAgent || hoveredAgent)!.vy).toFixed(2)}</span>
              {selectedAgent && (
                <button
                  onClick={() => setSelectedAgent(null)}
                  className="px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-sans text-xs"
                >
                  Deselect ✕
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Interactive Scenario & Physics Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Interactive In-Game Dynamic Scenarios */}
        <div className="lg:col-span-2 card p-4 border border-zinc-800 bg-zinc-900/70">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
              <span>⚡</span> Trigger Match Event In-Game Shocks
            </span>
            <span className="text-[10px] text-zinc-500">Watch the swarm instantaneously reconfigure</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {[
              { id: "equilibrium", label: "Equilibrium", icon: "⚖️" },
              { id: "home_goal", label: "Home Goal 1-0", icon: "⚽" },
              { id: "red_card_home", label: "Home Red Card", icon: "🟥" },
              { id: "muddy_pitch", label: "Heavy Mud Pitch", icon: "🌧️" },
              { id: "sharp_steam", label: "Sharp Money Shift", icon: "📈" },
            ].map((scen) => (
              <button
                key={scen.id}
                onClick={() => applyScenario(scen.id)}
                className={`px-3 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  activeScenario === scen.id
                    ? "bg-purple-600 text-white shadow-md shadow-purple-600/30 border border-purple-400"
                    : "bg-zinc-950 text-zinc-300 hover:bg-zinc-800 border border-zinc-800"
                }`}
              >
                <span>{scen.icon}</span>
                <span className="truncate">{scen.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Physics Tuning Controls */}
        <div className="card p-4 border border-zinc-800 bg-zinc-900/70 space-y-3 text-xs">
          <span className="font-bold text-white uppercase tracking-wider block">
            Swarm Physics Controls
          </span>

          <div className="space-y-2">
            <div className="flex justify-between text-zinc-400">
              <span>Flocking Cohesion:</span>
              <span className="font-mono text-purple-300 font-bold">{(cohesionStrength * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="1.5"
              step="0.1"
              value={cohesionStrength}
              onChange={(e) => setCohesionStrength(Number(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
          </div>

          <div className="flex justify-between items-center pt-1 border-t border-zinc-800">
            <span className="text-zinc-400">Agent Population:</span>
            <div className="flex gap-1.5">
              {[80, 140, 220].map((count) => (
                <button
                  key={count}
                  onClick={() => setParticleCount(count)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition-colors ${
                    particleCount === count
                      ? "bg-purple-600 text-white"
                      : "bg-zinc-950 text-zinc-400 hover:text-zinc-200 border border-zinc-800"
                  }`}
                >
                  {count}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Canvas Drawing Helper Functions
// ---------------------------------------------------------------------------

function drawTacticalPitch(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.save();
  ctx.strokeStyle = "rgba(39, 39, 42, 0.4)";
  ctx.lineWidth = 1;

  // Outer border pitch
  const m = 30;
  ctx.strokeRect(m, m, w - m * 2, h - m * 2);

  // Center line
  ctx.beginPath();
  ctx.moveTo(w / 2, m);
  ctx.lineTo(w / 2, h - m);
  ctx.stroke();

  // Center circle
  ctx.beginPath();
  ctx.arc(w / 2, h / 2, 70, 0, Math.PI * 2);
  ctx.stroke();

  // Left penalty box (Home)
  ctx.strokeRect(m, h * 0.3, 110, h * 0.4);
  // Right penalty box (Away)
  ctx.strokeRect(w - m - 110, h * 0.3, 110, h * 0.4);

  ctx.restore();
}

function drawAttractor(ctx: CanvasRenderingContext2D, att: Attractor, now: number) {
  ctx.save();

  // Pulsing outer halo
  const pulse = Math.sin(now * 0.003 + (att.key === "H" ? 0 : att.key === "D" ? 2 : 4)) * 6;
  const outerRadius = att.radius + 12 + pulse;

  const grad = ctx.createRadialGradient(att.x, att.y, att.radius * 0.3, att.x, att.y, outerRadius);
  grad.addColorStop(0, att.glowColor);
  grad.addColorStop(1, "rgba(0,0,0,0)");

  ctx.beginPath();
  ctx.arc(att.x, att.y, outerRadius, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Outer border ring
  ctx.beginPath();
  ctx.arc(att.x, att.y, att.radius + 4, 0, Math.PI * 2);
  ctx.strokeStyle = att.color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Inner core disc
  ctx.beginPath();
  ctx.arc(att.x, att.y, att.radius, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(18, 18, 23, 0.95)";
  ctx.fill();

  // Attractor text label
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "bold 11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(att.key === "H" ? "HOME" : att.key === "D" ? "DRAW" : "AWAY", att.x, att.y - 7);

  ctx.fillStyle = att.color;
  ctx.font = "bold 13px monospace";
  ctx.fillText(`${(att.targetProb * 100).toFixed(0)}%`, att.x, att.y + 9);

  // Subtitle team label under attractor
  ctx.fillStyle = "#A1A1AA";
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText(att.teamLabel, att.x, att.y + att.radius + 18);

  ctx.restore();
}

function drawSynapticLinks(ctx: CanvasRenderingContext2D, agents: Agent[]) {
  ctx.save();
  ctx.lineWidth = 0.6;

  for (let i = 0; i < agents.length; i++) {
    for (let j = i + 1; j < agents.length; j++) {
      const a = agents[i];
      const b = agents[j];

      // Only link if different personas (cross-agent debate) and nearby
      if (a.persona !== b.persona) {
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 50) {
          const alpha = (1 - d / 50) * 0.25;
          ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
  }
  ctx.restore();
}

function drawAgent(ctx: CanvasRenderingContext2D, a: Agent, isHighlighted: boolean) {
  ctx.save();

  // Draw motion trail
  if (a.trail.length > 1) {
    ctx.beginPath();
    ctx.moveTo(a.trail[0].x, a.trail[0].y);
    for (let i = 1; i < a.trail.length; i++) {
      ctx.lineTo(a.trail[i].x, a.trail[i].y);
    }
    ctx.strokeStyle = `${a.color}30`;
    ctx.lineWidth = a.radius * 0.75;
    ctx.stroke();
  }

  // Highlight target ring
  if (isHighlighted) {
    ctx.beginPath();
    ctx.arc(a.x, a.y, a.radius + 7, 0, Math.PI * 2);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Core Agent Node
  ctx.beginPath();
  ctx.arc(a.x, a.y, isHighlighted ? a.radius + 2 : a.radius, 0, Math.PI * 2);
  ctx.fillStyle = a.color;
  ctx.shadowColor = a.color;
  ctx.shadowBlur = isHighlighted ? 12 : 6;
  ctx.fill();

  // Heading indicator line
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(a.x + a.vx * 2.5, a.y + a.vy * 2.5);
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.restore();
}
