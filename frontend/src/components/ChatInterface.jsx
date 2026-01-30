import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useNavigate } from 'react-router-dom';
import { inferAdaptive, STREAM_ENDPOINT } from '../api';
import './ChatInterface.css';

const API_BASE_URL = 'http://localhost:8000';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { role: 'system', content: 'FastAPI Inference System Online. Ready for queries.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState('dark');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
        if (e.key === '/' && document.activeElement !== inputRef.current) {
            e.preventDefault();
            inputRef.current?.focus();
        }
        if (e.key === 'Escape') {
            inputRef.current?.blur();
        }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Theme Toggle Effect
  useEffect(() => {
      document.body.className = theme === 'dark' ? 'dark-theme' : 'light-theme';
  }, [theme]);

  const toggleTheme = () => {
      setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const copyToClipboard = async (text) => {
      try {
          await navigator.clipboard.writeText(text);
          // Optional: Show toast
      } catch (err) {
          console.error('Failed to copy:', err);
      }
  };

  const submitFeedback = async (msgIndex, label) => {
      const msg = messages[msgIndex];
      if (msg.role !== 'assistant') return;
      
      // OPTIMISTIC UI UPDATE: Update state immediately before API call
      setMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[msgIndex] = { 
              ...newMsgs[msgIndex], 
              feedback: label  // 'correct', 'incorrect', or 'should_have_refused'
          };
          return newMsgs;
      });
      
      // Find query (scan backwards)
      let query = '';
      for (let i = msgIndex - 1; i >= 0; i--) {
          if (messages[i].role === 'user') {
              query = messages[i].content;
              break;
          }
      }
      
      if (!query) return; 
      
      try {
          await fetch(`${API_BASE_URL}/feedback`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  query: query,
                  response: msg.content,
                  label: label,  // New schema: 'correct', 'incorrect', 'should_have_refused'
                  model_mode: msg.meta?.mode || 'adaptive',
                  confidence: msg.meta?.confidence || null
              })
          });
      } catch (err) {
          console.error("Feedback failed:", err);
          // Revert on error
          setMessages(prev => {
            const newMsgs = [...prev];
            if (newMsgs[msgIndex]) {
                 const updated = { ...newMsgs[msgIndex] };
                 delete updated.feedback;
                 newMsgs[msgIndex] = updated;
            }
            return newMsgs;
          });
      }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const data = await inferAdaptive(input);
      console.log('API Response:', data); // Debug log
      
      if (!data || typeof data.answer !== 'string') {
        throw new Error('Invalid response structure: missing answer');
      }
      
      const assistantMsg = { 
        role: 'assistant', 
        content: data.answer,
        meta: {
          mode: data.intent || 'adaptive',
          source: data.source || 'unknown',
          confidence: data.confidence,
          refused: !!data.refused,
          citations: Array.isArray(data.citations) ? data.citations : [],
          context_used: data.source === 'rag'
        }
      };
      
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error('Frontend error:', error);
      setMessages(prev => [...prev, { role: 'error', content: `Error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`chat-container ${theme}`}>
      <div className="chat-header">
        <h2>FastAPI LLM Inference</h2>
      </div>

      <div className="messages-area">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className={`message-bubble ${msg.role}`}>
              {msg.meta && (
                <div className={`meta-badge ${msg.meta.refused ? 'refused' : ''}`}>
                  <span className="source-tag">{msg.meta.source ? msg.meta.source.toUpperCase() : 'UNKNOWN'}</span>
                  {msg.meta.confidence !== undefined && (
                    <span className="confidence-tag">
                      {Math.round(msg.meta.confidence * 100)}%
                    </span>
                  )}
                </div>
              )}
              
              <ReactMarkdown
                components={{
                  code({node, inline, className, children, ...props}) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        customStyle={{ 
                            margin: 0, 
                            borderRadius: '8px', 
                            background: '#1e1e1e' 
                        }}
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    )
                  }
                }}
              >
                {msg.content}
              </ReactMarkdown>

              {/* Citations Section */}
              {msg.meta?.citations?.length > 0 && (
                <div className="citations-section" style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#6b7280', marginBottom: '8px', letterSpacing: '0.05em' }}>
                    Sources
                  </div>
                  <div className="citations-grid" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {msg.meta.citations.map((cite, i) => (
                      <a 
                        key={i}
                        href={cite.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="citation-chip"
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          background: 'rgba(255, 255, 255, 0.05)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          color: '#60a5fa',
                          fontSize: '0.85rem',
                          textDecoration: 'none',
                          maxWidth: '100%',
                          transition: 'background 0.2s'
                        }}
                        onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
                        onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'}
                      >
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '300px' }}>
                          {cite.title || cite.url}
                        </span>
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Message Actions */}
              {msg.role === 'assistant' && (
                  <div className="message-actions">
                      <button 
                          onClick={() => copyToClipboard(msg.content)} 
                          data-tooltip="Copy"
                          className="action-btn"
                      >
                          {/* Copy Icon */}
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                      </button>
                      <button 
                          onClick={() => submitFeedback(index, 'correct')} 
                          className={`action-btn ${msg.feedback === 'correct' ? 'active' : ''}`}
                          data-tooltip="Helpful"
                          disabled={!!msg.feedback || msg.meta?.refused}
                          style={(!!msg.feedback || msg.meta?.refused) ? { cursor: 'not-allowed', opacity: msg.feedback === 'correct' ? 1 : 0.3 } : {}}
                      >
                          {/* Thumbs Up */}
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                          </svg>
                      </button>
                      <button 
                          onClick={() => submitFeedback(index, 'incorrect')} 
                          className={`action-btn ${msg.feedback === 'incorrect' ? 'active' : ''}`}
                          data-tooltip="Not Helpful"
                          disabled={!!msg.feedback || msg.meta?.refused}
                          style={(!!msg.feedback || msg.meta?.refused) ? { cursor: 'not-allowed', opacity: msg.feedback === 'incorrect' ? 1 : 0.3 } : {}}
                      >
                          {/* Thumbs Down */}
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
                          </svg>
                      </button>
                  </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="system-message left-align">
                 <div className="loading">Thinking</div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-row">
          <form className="input-area" onSubmit={handleSubmit}>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type / to focus..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading}>Send</button>
          </form>
      </div>

      <div className="chat-controls">
        <button className="health-btn" onClick={() => navigate('/health')} title="View System Health Stats">
            <span className="health-dot"></span> Health
        </button>
      </div>

      <div className="theme-controls">
        <button className="theme-toggle" onClick={toggleTheme} title="Toggle Day/Night Mode">
            {theme === 'dark' ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="5"></circle>
                    <line x1="12" y1="1" x2="12" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="23"></line>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                    <line x1="1" y1="12" x2="3" y2="12"></line>
                    <line x1="21" y1="12" x2="23" y2="12"></line>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                </svg>
            ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                </svg>
            )}
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
