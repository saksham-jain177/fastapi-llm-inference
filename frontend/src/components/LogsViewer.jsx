import React, { useState, useEffect } from 'react';
import './LogsViewer.css';

const LogsViewer = ({ isOpen, onClose }) => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState(null);

    useEffect(() => {
        if (isOpen) {
            fetchLogs();
        }
    }, [isOpen]);

    const fetchLogs = async () => {
        try {
            const response = await fetch('http://localhost:8000/logs/recent');
            const data = await response.json();
            setStats({
                total: data.total_count,
                source: data.source,
                redis: data.redis_status
            });
            setLogs(data.recent_logs);
            setLoading(false);
        } catch (err) {
            console.error('Failed to fetch logs:', err);
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="logs-overlay" onClick={onClose}>
            <div className="logs-panel" onClick={(e) => e.stopPropagation()}>
                <div className="logs-header">
                    <h2>Feedback Logs</h2>
                    <button onClick={onClose} className="close-btn">×</button>
                </div>

                <div className="logs-status">
                    <div className={`status-badge ${stats?.source === 'mongo' ? 'success' : 'warning'}`}>
                        MongoDB: {stats?.source === 'mongo' ? '✓ Connected' : '⚠ File Fallback'}
                    </div>
                    <div className={`status-badge ${stats?.redis === 'connected' ? 'success' : 'danger'}`}>
                        Redis: {stats?.redis === 'connected' ? '✓ Connected' : '✗ Disconnected'}
                    </div>
                    <div className="status-count">
                        Total Logs: {stats?.total || 0}
                    </div>
                </div>

                <div className="logs-list">
                    {loading ? (
                        <div className="loading-text">Loading...</div>
                    ) : logs.length === 0 ? (
                        <div className="empty-text">No feedback logs yet. Start voting! 👍👎</div>
                    ) : (
                        logs.map((log, idx) => (
                            <div key={idx} className={`log-entry ${log.feedback === '1' ? 'positive' : log.feedback === '-1' ? 'negative' : 'neutral'}`}>
                                <div className="log-header">
                                    <span className="log-time">{new Date(log.timestamp).toLocaleString()}</span>
                                    <span className={`log-rating ${log.feedback === '1' ? 'up' : 'down'}`}>
                                        {log.feedback === '1' ? (
                                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#4CAF50" stroke="currentColor" strokeWidth="2">
                                                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                                            </svg>
                                        ) : (
                                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#F44336" stroke="currentColor" strokeWidth="2">
                                                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
                                            </svg>
                                        )}
                                    </span>
                                </div>
                                <div className="log-query">Q: {log.query}</div>
                                <div className="log-response">A: {log.response.substring(0, 100)}...</div>
                                <div className="log-meta">Intent: {log.intent}</div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default LogsViewer;
