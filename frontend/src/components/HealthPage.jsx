
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchSystemStats } from '../api';
import './HealthPage.css';

const HealthPage = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                navigate('/');
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [navigate]);

    useEffect(() => {
        const loadStats = async () => {
            try {
                const data = await fetchSystemStats();
                setStats(data);
                setError(null);
            } catch (err) {
                console.error("Health fetch error:", err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadStats();
        // Refresh every 3 seconds for "live" feel
        const interval = setInterval(loadStats, 3000);
        return () => clearInterval(interval);
    }, []);

    // Derived Metrics
    const calculateStatus = () => {
        if (!stats) return { label: 'Unknown', color: 'gray' };
        if (stats.active_requests > 50) return { label: 'High Load', color: 'orange' };
        if (stats.total_errors > 0 && (stats.total_errors / stats.total_requests) > 0.05) {
            return { label: 'Degraded', color: 'red' };
        }
        return { label: 'Operational', color: '#4CAF50' };
    };

    if (loading && !stats) return (
        <div className="health-page">
            <div className="loading-spinner">Loading System Telemetry...</div>
        </div>
    );

    if (error) return (
        <div className="health-page">
             <div className="error-container">
                <h3>Connection Failure</h3>
                <p>{error}</p>
                <button onClick={() => navigate('/')} className="back-btn">Return to Safety</button>
             </div>
        </div>
    );

    const status = calculateStatus();
    const hitRate = stats.total_requests > 0 
        ? ((stats.cache_hits / (stats.cache_hits + stats.cache_misses || 1)) * 100).toFixed(1) 
        : 0;
    const errorRate = stats.total_requests > 0
        ? ((stats.total_errors / stats.total_requests) * 100).toFixed(1)
        : 0;

    return (
        <div className="health-page">
            <div className="health-container">
                {/* Header */}
                <div className="health-header">
                    <div className="title-group">
                        <h2>System Status</h2>
                        <span className="status-badge" style={{ backgroundColor: status.color }}>
                            {status.label}
                        </span>
                    </div>
                    <button onClick={() => navigate('/')} className="back-btn">
                        Exit Dashboard (Esc)
                    </button>
                </div>

                {/* Primary Metrics */}
                <div className="metrics-grid">
                    <div className="metric-card">
                        <div className="meta">
                            <label>Active Requests</label>
                            <div className="value">{stats.active_requests}</div>
                        </div>
                    </div>
                    
                    <div className="metric-card">
                        <div className="meta">
                            <label>Total Inferences</label>
                            <div className="value">{stats.total_requests}</div>
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="meta">
                            <label>Cache Hit Rate</label>
                            <div className="value">{hitRate}%</div>
                            <div className="progress-bar">
                                <div className="fill" style={{ width: `${hitRate}%`, '--bar-color': '#2196F3' }}></div>
                            </div>
                        </div>
                    </div>

                    <div className="metric-card">
                        <div className="meta">
                            <label>Error Rate</label>
                            <div className="value">{errorRate}%</div>
                            <div className="progress-bar">
                                <div className="fill" style={{ width: `${errorRate}%`, '--bar-color': '#ff5252' }}></div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Detailed Logs / Adapter Usage */}
                <div className="dashboard-row">
                    <div className="dashboard-panel">
                        <h4>Domain Adapter Usage</h4>
                        {stats.adapter_usage && Object.keys(stats.adapter_usage).length > 0 ? (
                            <ul className="adapter-list">
                                {Object.entries(stats.adapter_usage).map(([domain, count]) => (
                                    <li key={domain}>
                                        <span className="domain-id">{domain}</span>
                                        <span className="usage-count">{count} reqs</span>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <div className="empty-state">No domain-specific routing detected yet.</div>
                        )}
                    </div>

                    <div className="dashboard-panel info-panel">
                        <h4>System Info</h4>
                        <div className="info-row">
                            <span>Backend</span>
                            <span>FastAPI v2.0</span>
                        </div>
                        <div className="info-row">
                            <span>Model</span>
                            <span>Quantized LLaMA (4-bit)</span>
                        </div>
                        <div className="info-row">
                            <span>Environment</span>
                            <span>Production (Local)</span>
                        </div>
                        <div className="info-row">
                            <span>Monitoring</span>
                            <a href="http://localhost:9090" target="_blank" rel="noopener noreferrer" className="prometheus-link">
                                Open Prometheus ↗
                            </a>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default HealthPage;
