import React, { useState } from 'react';
import axios from 'axios';
import Sidebar from './Sidebar';
import './App.css';
import { SignedIn, SignedOut, SignInButton, UserButton, useAuth } from "@clerk/clerk-react";
import { FaCopy, FaCheck } from 'react-icons/fa';

function App() {
  const [repo, setRepo] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [copied, setCopied] = useState(false);
  
  // Sidebar State
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [jobSteps, setJobSteps] = useState([]);
  const [runUrl, setRunUrl] = useState("");
  const [finalStatus, setFinalStatus] = useState("");

  const { getToken } = useAuth(); 

  // --- HANDLERS ---
  const handleFileChange = (e) => {
    if (e.target.files) setFile(e.target.files[0]);
  };

  const copyToClipboard = () => {
    const code = `name: Train My Model
on: workflow_dispatch

jobs:
  call-engine:
    uses: KariraLakshya/mlops-auto-trainer/.github/workflows/main_logic.yml@main`;
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleTrigger = async () => {
    if (!file || !repo) return alert("Missing file or repo!");

    setLoading(true);
    setMsg("🚀 Authenticating & Uploading...");
    setIsSidebarOpen(true);
    setJobSteps([]);

    const formData = new FormData();
    formData.append('file', file);
    const [owner, repoName] = repo.split('/');
    formData.append('owner', owner);
    formData.append('repo', repoName);

    try {
      const sessionToken = await getToken();
      const res = await axios.post('http://localhost:3000/api/trigger', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${sessionToken}`
        }
      });

      setMsg("✅ Triggered! ID: " + res.data.runId);
      pollStatus(res.data.runId, sessionToken);

    } catch (err) {
      console.error(err);
      setMsg("❌ Error: " + (err.response?.data?.error || err.message));
      setLoading(false);
    }
  };

  const pollStatus = (runId, token) => {
    const [owner, repoName] = repo.split('/');
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`http://localhost:3000/api/status/${runId}`, {
          params: { owner, repo: repoName },
          headers: { 'Authorization': `Bearer ${token}` }
        });

        const { status, steps, html_url, conclusion } = res.data;
        setJobSteps(steps || []);
        setRunUrl(html_url);
        setFinalStatus(status);

        if (status === 'completed') {
          clearInterval(interval);
          setLoading(false);
          setMsg(conclusion === 'success' ? "🎉 Success!" : "⚠️ Failed.");
        }
      } catch (err) {
        if (err.response && (err.response.status === 401)) clearInterval(interval);
      }
    }, 2000);
  };

  return (
    <div>
      {/* HEADER */}
      <header style={{position: 'absolute', top: 20, right: 30, zIndex: 50}}>
        <SignedIn>
          <UserButton afterSignOutUrl="/" />
        </SignedIn>
      </header>

      {/* --- 1. SIGNED OUT VIEW (THE INSTRUCTIONS) --- */}
      <SignedOut>
        <div className="container" style={{ width: '600px', textAlign: 'left' }}>
          <h1>🚀 MLOps Platform</h1>
          <p className="subtitle">Connect your data to our Universal Training Engine.</p>

          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px', marginBottom: '20px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <h3 style={{marginTop: 0, color: '#fff'}}>⚙️ First Time Setup</h3>
            <p style={{fontSize: '0.9rem', color: '#ccc'}}>
              To use this platform, create a file in your repository at: <br/>
              <code style={{color: '#3b82f6'}}>.github/workflows/user_trigger.yml</code>
            </p>
            
            {/* CODE BLOCK */}
            <div style={{ position: 'relative', background: '#0f172a', padding: '15px', borderRadius: '8px', border: '1px solid #334155', fontFamily: 'monospace', fontSize: '0.85rem', color: '#a5b4fc', overflowX: 'auto' }}>
              <pre style={{margin: 0}}>
{`name: Train My Model
on: workflow_dispatch

jobs:
  call-engine:
    uses: KariraLakshya/mlops-auto-trainer/.github/workflows/main_logic.yml@main`}
              </pre>
              <button 
                onClick={copyToClipboard}
                style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(255,255,255,0.1)', border: 'none', color: '#fff', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer' }}
              >
                {copied ? <FaCheck color="#10b981"/> : <FaCopy />}
              </button>
            </div>
            <p style={{fontSize: '0.8rem', color: '#64748b', marginTop: '10px', fontStyle: 'italic'}}>
              This connects your data to our training engine automatically.
            </p>
          </div>

          <SignInButton mode="modal">
            <button className="btn-primary" style={{ marginTop: '0' }}>Sign In & Launch</button>
          </SignInButton>
        </div>
      </SignedOut>

      {/* --- 2. SIGNED IN VIEW (THE DASHBOARD) --- */}
      <SignedIn>
        <div className="container">
          <h1>🚀 Mission Control</h1>
          <p className="subtitle">Universal Training Interface</p>

            <div className="input-group">
              <label>GitHub Repository</label>
              <input 
                type="text" 
                value={repo} 
                onChange={(e) => setRepo(e.target.value)} 
                placeholder="Owner/Repo (e.g. Dave/my-data)" 
              />
            </div>

            <div className="input-group">
              <label>Training Data (CSV)</label>
              <div className="file-drop">
                <input type="file" onChange={handleFileChange} accept=".csv" />
                <p>
                   {file ? <span className="file-name">📄 {file.name}</span> : "Click to Upload Dataset"}
                </p>
              </div>
            </div>

            <button className="btn-primary" onClick={handleTrigger} disabled={loading}>
              {loading ? "Processing..." : "🚀 Launch Training"}
            </button>
            <p className="status-msg">{msg}</p>
        </div>
      </SignedIn>

      <Sidebar isOpen={isSidebarOpen} steps={jobSteps} runUrl={runUrl} status={finalStatus} />
    </div>
  );
}

export default App;