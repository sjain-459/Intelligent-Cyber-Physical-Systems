import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, ShieldAlert, Cpu, Network, Play, ShieldCheck, Database, GitFork, Radio } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

// The backend may be served from a different origin in production (e.g.
// separate Render services for frontend/backend). VITE_API_BASE_URL (set at
// build time) overrides the localhost default used for local development.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

// DEMO_MODE replays a faithful recording of one real pipeline run
// (frontend/public/demo_run.json) instead of opening a live WebSocket.
// Used for the public demo deploy, which has no backend: running the real
// federated-learning + differential-privacy + SHAP stack live, per visitor,
// needs more RAM than a free hosting tier provides. Local development
// (`python api.py` + `npm run dev`) is unaffected and always runs live.
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

const STATUS_COPY = {
  idle: 'Ready',
  starting: 'Connecting…',
  training: 'Federated training in progress',
  monitoring: 'Scanning process stages for anomalies',
  alert: 'Threat detected',
  completed: 'Run complete',
};

export default function App() {
  const [status, setStatus] = useState('idle'); // idle, starting, training, monitoring, alert, completed
  const [messages, setMessages] = useState([]);

  // Training state
  const [currentRound, setCurrentRound] = useState(0);
  const [clientStatus, setClientStatus] = useState({});
  const [trainingMetrics, setTrainingMetrics] = useState([]);
  const [metricsImage, setMetricsImage] = useState(null);

  // Threat detection state
  const [targetStage, setTargetStage] = useState(null);
  const [threshold, setThreshold] = useState(0);
  const [sensorStream, setSensorStream] = useState([]);
  const [anomalyIndex, setAnomalyIndex] = useState(null);
  const [xaiData, setXaiData] = useState(null);

  const messagesEndRef = useRef(null);
  const socketRef = useRef(null); // Ref to hold the WebSocket instance
  const replayRef = useRef({ cancelled: false }); // Cancels an in-flight replay on restart/unmount

  const hasStarted = status !== 'idle';

  // Auto-scroll for logs
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Clean up any live connection / in-flight replay on unmount
  useEffect(() => {
    const replayState = replayRef.current;
    return () => {
      if (socketRef.current) socketRef.current.close();
      replayState.cancelled = true;
    };
  }, []);

  const handleEvent = useCallback((data) => {
    const ev = data.event;

    if (ev === 'init' || ev === 'info') {
      setMessages((p) => [...p, data.message]);
    }
    if (ev === 'round_start') {
      setStatus('training');
      setCurrentRound(data.round);
    }
    if (ev === 'client_training') {
      setClientStatus((p) => ({ ...p, [data.client_id]: data.status }));
    }
    if (ev === 'round_end') {
      setClientStatus({});
      setTrainingMetrics((p) => [
        ...p,
        {
          round: data.round,
          loss: data.loss.toFixed(4),
          epsilon: data.epsilon.toFixed(2),
          trusted: data.metrics.trusted_clients ?? 6,
        },
      ]);
    }
    if (ev === 'fl_done') {
      setMessages((p) => [...p, 'Federated learning completed.']);
      setMetricsImage(
        DEMO_MODE
          ? '/federated_metrics.png'
          : `${API_BASE_URL}/results/federated_metrics.png?t=${Date.now()}`
      );
    }
    if (ev === 'threat_detect_start') {
      setStatus('monitoring');
      setTargetStage(data.target_stage);
    }
    if (ev === 'threshold_computed') {
      setThreshold(data.threshold);
    }
    if (ev === 'sensor_stream') {
      setSensorStream((p) => {
        const newEntry = { index: data.index, error: data.error, ewma: data.ewma };
        const newStream = [...p, newEntry];
        return newStream.length > 50 ? newStream.slice(-50) : newStream;
      });
    }
    if (ev === 'anomaly_detected') {
      setStatus('alert');
      setAnomalyIndex(data.index);
    }
    if (ev === 'xai_results') {
      setXaiData(data);
    }
    if (ev === 'done') {
      setStatus((prev) => (prev === 'alert' ? 'alert' : 'completed'));
    }
  }, []);

  const resetState = () => {
    setMessages([]);
    setCurrentRound(0);
    setClientStatus({});
    setTrainingMetrics([]);
    setSensorStream([]);
    setAnomalyIndex(null);
    setXaiData(null);
    setMetricsImage(null);
    setThreshold(0);
    setTargetStage(null);
  };

  const runReplay = async () => {
    replayRef.current.cancelled = false;
    try {
      const res = await fetch('/demo_run.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const events = await res.json();

      for (const data of events) {
        if (replayRef.current.cancelled) return;
        handleEvent(data);

        let delay = 200;
        if (data.event === 'client_training' && data.status === 'training') delay = 400;
        else if (data.event === 'sensor_stream') delay = 45;
        await new Promise((r) => setTimeout(r, delay));
      }
    } catch (err) {
      setMessages((p) => [...p, `Failed to load the recorded demo run: ${err.message}`]);
      setStatus('idle');
    }
  };

  const startSimulation = () => {
    if (socketRef.current) socketRef.current.close();
    replayRef.current.cancelled = true; // cancel any prior in-flight replay

    resetState();
    setStatus('starting');

    if (DEMO_MODE) {
      runReplay();
      return;
    }

    const ws = new WebSocket(`${WS_BASE_URL}/ws/simulation`);

    ws.onmessage = (event) => {
      try {
        handleEvent(JSON.parse(event.data));
      } catch (err) {
        console.error('Failed to parse WS message', err);
      }
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === 'completed' || prev === 'alert' ? prev : 'idle'));
    };

    ws.onerror = () => {
      setMessages((p) => [...p, 'Connection error: backend unreachable.']);
      setStatus('idle');
    };

    socketRef.current = ws;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex flex-col">
      <div className="flex-1 p-4 sm:p-6 max-w-[1400px] w-full mx-auto">
        <header className="flex flex-wrap items-center justify-between gap-4 mb-8 border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30 shrink-0">
              <ShieldCheck className="w-7 h-7 sm:w-8 sm:h-8 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  CTMAS
                </h1>
                {DEMO_MODE && (
                  <span className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-full px-2 py-0.5">
                    <Radio className="w-3 h-3" /> Recorded demo
                  </span>
                )}
              </div>
              <p className="text-xs sm:text-sm text-slate-400">Proactive Threat Modeling for Cyber-Physical Systems</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {(status === 'idle' || status === 'completed') ? (
              <button
                onClick={startSimulation}
                className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-2.5 rounded-full font-medium transition-all shadow-[0_0_15px_rgba(6,182,212,0.4)] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              >
                <Play className="w-4 h-4" /> {status === 'completed' ? 'Run again' : 'Start simulation'}
              </button>
            ) : (
              <div
                role="status"
                className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
                  status === 'alert'
                    ? 'bg-red-500/10 border-red-500/50 text-red-400'
                    : 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
                }`}
              >
                <Activity className={`w-4 h-4 ${status === 'training' || status === 'monitoring' ? 'animate-pulse' : ''}`} />
                <span className="font-medium text-sm">{STATUS_COPY[status]}</span>
              </div>
            )}
          </div>
        </header>

        {!hasStarted && (
          <section className="mb-8 bg-slate-900/40 border border-white/10 rounded-2xl p-6">
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed max-w-3xl">
              CTMAS trains an anomaly-detection model across six water-treatment process stages
              (P1–P6) without any stage sharing raw sensor data — each stage trains locally under{' '}
              <span className="text-cyan-400 font-medium">differential privacy</span> and only
              shares model updates, aggregated by a{' '}
              <span className="text-cyan-400 font-medium">trust-aware</span> server that rejects
              poisoned updates. The trained model then scans all six stages for anomalies,
              explains what it found with{' '}
              <span className="text-cyan-400 font-medium">SHAP</span>, and maps it to{' '}
              <span className="text-cyan-400 font-medium">STRIDE</span> /{' '}
              <span className="text-cyan-400 font-medium">MITRE ATT&amp;CK for ICS</span> threat
              categories.
            </p>
            {DEMO_MODE && (
              <p className="mt-3 text-xs text-amber-300/80 max-w-3xl">
                This is a faithful replay of one real run against the real SWaT dataset — not a
                fresh computation per visitor. Running the real federated-learning stack live,
                per visitor, needs more memory than a free public demo can afford; the
                self-hosted version runs it live end-to-end.
              </p>
            )}
          </section>
        )}

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: FL Training & Logs */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900/40 border border-white/10 p-6 rounded-2xl relative overflow-hidden">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-white">
                <Network className="w-5 h-5 text-cyan-400" /> Federated Learning Swarm
              </h2>

              <div className="grid grid-cols-3 gap-4 mb-8">
                <MetricCard label="Global Round" value={`${currentRound} / 5`} color="text-cyan-400" />
                <MetricCard
                  label="Privacy (ε)"
                  value={trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1].epsilon : '0.00'}
                  color="text-amber-400"
                />
                <MetricCard
                  label="Global Loss"
                  value={trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1].loss : '0.00'}
                  color="text-green-400"
                />
              </div>

              <span className="text-sm font-medium text-slate-300 block mb-4">Process stages (P1–P6)</span>
              <div className="grid grid-cols-3 gap-4">
                {[1, 2, 3, 4, 5, 6].map((id) => (
                  <div
                    key={id}
                    className={`p-4 rounded-xl border transition-all duration-500 ${
                      clientStatus[id] === 'training'
                        ? 'bg-cyan-900/40 border-cyan-500/50 scale-105 shadow-lg'
                        : 'bg-slate-800/50 border-white/5'
                    }`}
                  >
                    <Cpu className={`w-8 h-8 mb-2 mx-auto ${clientStatus[id] === 'training' ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
                    <p className="text-center text-xs font-medium">Stage P{id}</p>
                  </div>
                ))}
              </div>
            </div>

            {metricsImage && (
              <div className="bg-slate-900/40 border border-cyan-500/30 p-4 rounded-2xl animate-in fade-in zoom-in duration-500">
                <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <Activity className="w-3 h-3" /> Training Analytics
                </h3>
                <img
                  src={metricsImage}
                  alt="Federated learning loss and privacy budget over training rounds"
                  className="w-full rounded-lg border border-white/5 shadow-2xl"
                />
              </div>
            )}

            <div className="bg-slate-900/60 border border-white/10 rounded-2xl flex flex-col h-64">
              <div className="p-3 border-b border-white/10 bg-slate-900/50 flex items-center gap-2">
                <Database className="w-4 h-4 text-slate-400" />
                <span className="text-xs font-mono text-slate-400">System Telemetry</span>
              </div>
              <div className="p-4 overflow-y-auto flex-1 font-mono text-xs space-y-2">
                {messages.length === 0 && (
                  <p className="text-slate-600">Telemetry will appear here once the run starts.</p>
                )}
                {messages.map((msg, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-cyan-500">{'>'}</span>
                    <span className={msg.includes('CRITICAL') ? 'text-red-400' : 'text-slate-300'}>{msg}</span>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            </div>
          </div>

          {/* Right Column: Threat Detection */}
          <div className="lg:col-span-7 space-y-6">
            <div
              className={`bg-slate-900/40 border p-6 rounded-2xl relative transition-colors duration-700 ${
                status === 'alert' ? 'border-red-500/50 bg-red-950/10' : 'border-white/10'
              }`}
            >
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-white">
                <Activity className={`w-5 h-5 ${status === 'alert' ? 'text-red-400' : 'text-cyan-400'}`} />
                Live Sensor Analysis {targetStage && `— Stage P${targetStage}`}
              </h2>

              <div className="h-72 sm:h-80 w-full relative min-h-[18rem]">
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                  <LineChart data={sensorStream}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="index" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                    {threshold > 0 && (
                      <ReferenceLine
                        y={threshold}
                        stroke="#ef4444"
                        strokeDasharray="3 3"
                        label={{ position: 'top', value: 'Threshold', fill: '#ef4444', fontSize: 10 }}
                      />
                    )}
                    <Line type="monotone" dataKey="error" name="Reconstruction error" stroke="#22d3ee" dot={false} strokeWidth={2} animationDuration={300} />
                    <Line type="monotone" dataKey="ewma" name="EWMA score" stroke="#f59e0b" dot={false} strokeWidth={2} animationDuration={300} />
                  </LineChart>
                </ResponsiveContainer>

                {(status === 'idle' || status === 'training' || status === 'starting') && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-950/90 backdrop-blur-sm rounded-xl">
                    <div className="w-20 h-20 rounded-full border border-cyan-500/30 flex items-center justify-center mb-4 relative">
                      <div className="absolute inset-0 border-2 border-cyan-500/20 rounded-full animate-ping" />
                      <ShieldCheck className="w-8 h-8 text-cyan-500/50" />
                    </div>
                    <p className="text-slate-400 text-sm animate-pulse text-center px-6">
                      {status === 'idle' && 'System armed — click "Start simulation" to begin'}
                      {status === 'starting' && (DEMO_MODE ? 'Loading recorded run…' : 'Connecting to backend…')}
                      {status === 'training' && 'Federated training in progress…'}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {xaiData && (
              <div className="bg-red-950/20 border border-red-500/40 p-6 rounded-2xl animate-in fade-in slide-in-from-bottom-4">
                <div className="flex items-center gap-3 mb-6 flex-wrap">
                  <ShieldAlert className="w-6 h-6 text-red-500 animate-pulse shrink-0" />
                  <h3 className="text-lg font-bold text-red-400">Explainable AI (SHAP) Diagnostics</h3>
                  {anomalyIndex !== null && (
                    <span className="ml-auto text-xs font-mono text-red-300/70">Window index: {anomalyIndex}</span>
                  )}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Compromised features</p>
                    <div className="flex flex-wrap gap-2">
                      {xaiData.features.map((f) => (
                        <span key={f} className="px-2 py-1 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-300 font-mono">
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-3">
                    {xaiData.alerts.map((alert, i) => (
                      <div key={i} className="bg-slate-900/80 p-3 rounded-lg border-l-4 border-red-500">
                        <div className="flex justify-between items-center mb-2 gap-2">
                          <span className="text-sm font-bold truncate">{alert['Affected Component']}</span>
                          <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded shrink-0">{alert['Stage']}</span>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 sm:gap-2 text-[10px] opacity-70">
                          <p>STRIDE: {alert['STRIDE Threat']}</p>
                          <p>MITRE: {alert['MITRE Class']}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      <footer className="border-t border-white/10 py-4 px-6 text-center text-xs text-slate-500">
        <a
          href="https://github.com/sjain-459/Intelligent-Cyber-Physical-Systems"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 hover:text-slate-300 transition-colors"
        >
          <GitFork className="w-3.5 h-3.5" /> Group 23 · The LNM Institute of Information Technology, Jaipur
        </a>
      </footer>
    </div>
  );
}

function MetricCard({ label, value, color }) {
  return (
    <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5">
      <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{label}</p>
      <p className={`text-xl font-mono font-bold ${color}`}>{value}</p>
    </div>
  );
}
