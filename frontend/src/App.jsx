import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, ShieldAlert, Cpu, Network, Server, Play, ShieldCheck, Database, Plus, X } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

// The backend may be served from a different origin in production (e.g.
// separate Render services for frontend/backend). VITE_API_BASE_URL (set at
// build time) overrides the localhost default used for local development.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

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

  // Dynamic node management
  const [nodes, setNodes] = useState([1, 2, 3, 4, 5, 6]);

  const messagesEndRef = useRef(null);
  const socketRef = useRef(null); // Ref to hold the WebSocket instance

  // Auto-scroll for logs
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, []);

  const addNode = () => {
    setNodes(prev => {
      const nextId = prev.length ? Math.max(...prev) + 1 : 1;
      return [...prev, nextId];
    });
  };

  const deleteNode = (id) => {
    setNodes(prev => prev.filter((n) => n !== id));
    setClientStatus((prev) => {
      const { [id]: _, ...rest } = prev;
      return rest;
    });
  };

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
          trusted: data.metrics.trusted_clients || 6,
        },
      ]);
    }
    if (ev === 'fl_done') {
      setMessages((p) => [...p, "Federated Learning completed."]);
      setMetricsImage(`${API_BASE_URL}/results/federated_metrics.png?t=${Date.now()}`);
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

  const startSimulation = () => {
    // 1. Clear previous connection if it exists
    if (socketRef.current) {
      socketRef.current.close();
    }

    // 2. Reset UI State
    setMessages([]);
    setCurrentRound(0);
    setClientStatus({});
    setTrainingMetrics([]);
    setSensorStream([]);
    setAnomalyIndex(null);
    setXaiData(null);
    setMetricsImage(null);
    setStatus('starting');

    // 3. Initialize WebSocket
    const ws = new WebSocket(`${WS_BASE_URL}/ws/simulation`);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEvent(data);
      } catch (err) {
        console.error("Failed to parse WS message", err);
      }
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === 'completed' || prev === 'alert' ? prev : 'idle'));
    };

    ws.onerror = () => {
      setMessages((p) => [...p, "Connection Error: Backend unreachable."]);
      setStatus('idle');
    };

    socketRef.current = ws;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 font-sans">
      <header className="flex items-center justify-between mb-8 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">CTMAS</h1>
            <p className="text-sm text-slate-400">Proactive Threat Modeling for Cyber-Physical Systems</p>
          </div>
        </div>
        <div>
          {status === 'idle' || status === 'completed' ? (
            <button
              onClick={startSimulation}
              className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white px-6 py-2.5 rounded-full font-medium transition-all shadow-[0_0_15px_rgba(6,182,212,0.4)]"
            >
              <Play className="w-4 h-4" /> {status === 'completed' ? 'Restart Simulation' : 'Start Simulation'}
            </button>
          ) : (
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
                status === 'alert' ? 'bg-red-500/10 border-red-500/50 text-red-400' : 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
              }`}>
              <Activity className={`w-4 h-4 ${(status === 'training' || status === 'monitoring') ? 'animate-pulse' : ''}`} />
              <span className="capitalize font-medium">{status.replace('_', ' ')}</span>
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: FL Training & Logs */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900/40 border border-white/10 p-6 rounded-2xl relative overflow-hidden">
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-white">
              <Network className="w-5 h-5 text-cyan-400" /> Federated Learning Swarm
            </h2>
            
            <div className="grid grid-cols-3 gap-4 mb-8">
              <MetricCard label="Global Round" value={`${currentRound} / 5`} color="text-cyan-400" />
              <MetricCard label="Privacy (ε)" value={trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1].epsilon : '0.00'} color="text-amber-400" />
              <MetricCard label="Global Loss" value={trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1].loss : '0.00'} color="text-green-400" />
            </div>

            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-slate-300">Active Nodes: {nodes.length}</span>
              <button onClick={addNode} className="flex items-center gap-1 bg-white/5 hover:bg-white/10 text-xs px-3 py-1 rounded border border-white/10 transition-colors">
                <Plus className="w-3 h-3" /> Add Node
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {nodes.map((id) => (
                <div key={id} className={`p-4 rounded-xl border relative group transition-all duration-500 ${
                  clientStatus[id] === 'training' ? 'bg-cyan-900/40 border-cyan-500/50 scale-105 shadow-lg' : 'bg-slate-800/50 border-white/5'
                }`}>
                  <button onClick={() => deleteNode(id)} className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 p-1">
                    <X className="w-3 h-3" />
                  </button>
                  <Cpu className={`w-8 h-8 mb-2 mx-auto ${clientStatus[id] === 'training' ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
                  <p className="text-center text-xs font-medium">Node {id}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Federated Metrics Plot Component */}
          {metricsImage && (
            <div className="bg-slate-900/40 border border-cyan-500/30 p-4 rounded-2xl animate-in fade-in zoom-in duration-500">
              <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <Activity className="w-3 h-3" /> Training Analytics
              </h3>
              <img 
                src={metricsImage} 
                alt="Federated Metrics" 
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
          <div className={`bg-slate-900/40 border p-6 rounded-2xl h-[450px] relative transition-colors duration-700 ${
            status === 'alert' ? 'border-red-500/50 bg-red-950/10' : 'border-white/10'
          }`}>
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-white">
              <Activity className={`w-5 h-5 ${status === 'alert' ? 'text-red-400' : 'text-cyan-400'}`} />
              Live Sensor Analysis {targetStage && `| Stage P${targetStage}`}
            </h2>

            <div className="h-80 w-full relative">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sensorStream}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="index" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                  {threshold > 0 && <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Threshold', fill: '#ef4444', fontSize: 10 }} />}
                  <Line type="monotone" dataKey="error" stroke="#22d3ee" dot={false} strokeWidth={2} animationDuration={300} />
                  <Line type="monotone" dataKey="ewma" stroke="#f59e0b" dot={false} strokeWidth={2} animationDuration={300} />
                </LineChart>
              </ResponsiveContainer>

              {/* Overlay for Idle/Training states */}
              {(status === 'idle' || status === 'training' || status === 'starting') && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-950/90 backdrop-blur-sm rounded-xl">
                  <div className="w-20 h-20 rounded-full border border-cyan-500/30 flex items-center justify-center mb-4 relative">
                    <div className="absolute inset-0 border-2 border-cyan-500/20 rounded-full animate-ping" />
                    <ShieldCheck className="w-8 h-8 text-cyan-500/50" />
                  </div>
                  <p className="text-slate-400 text-sm animate-pulse">
                    {status === 'training' ? 'Federated Training in progress...' : 'System Armed - Waiting for Data...'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* XAI Panel */}
          {xaiData && (
            <div className="bg-red-950/20 border border-red-500/40 p-6 rounded-2xl animate-in fade-in slide-in-from-bottom-4">
              <div className="flex items-center gap-3 mb-6">
                <ShieldAlert className="w-6 h-6 text-red-500 animate-pulse" />
                <h3 className="text-lg font-bold text-red-400">Explainable AI (SHAP) Diagnostics</h3>
                {anomalyIndex !== null && (
                  <span className="ml-auto text-xs font-mono text-red-300/70">Window Index: {anomalyIndex}</span>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Compromised Features</p>
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
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-bold">{alert['Affected Component']}</span>
                        <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded">{alert['Stage']}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-[10px] opacity-70">
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
      </div>
    </div>
  );
}

// Helper component for cleaner metric layout
function MetricCard({ label, value, color }) {
  return (
    <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5">
      <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{label}</p>
      <p className={`text-xl font-mono font-bold ${color}`}>{value}</p>
    </div>
  );
}