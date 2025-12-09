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
  const [mode, setMode] = useState('adaptive'); // 'adaptive' or 'stream'
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

  const submitFeedback = async (msgIndex, rating) => {
      const msg = messages[msgIndex];
      if (msg.role !== 'assistant') return;
      
      // Find the PREVIOUS user message (scan backwards)
      let query = '';
      for (let i = msgIndex - 1; i >= 0; i--) {
          if (messages[i].role === 'user') {
              query = messages[i].content;
              break;
          }
      }
      
      if (!query) return; // Should not happen in normal flow
      
      try {
          await fetch(`${API_BASE_URL}/feedback`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  query: query,
                  response: msg.content,
                  rating: rating,
                  model_mode: msg.meta?.mode || mode
              })
          });
          
          // Visual feedback update (optimistic UI)
          setMessages(prev => {
              const newMsgs = [...prev];
              newMsgs[msgIndex] = { 
                  ...newMsgs[msgIndex], 
                  feedback: rating === 1 ? 'up' : 'down' 
              };
              return newMsgs;
          });
      } catch (err) {
          console.error("Feedback failed:", err);
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
      if (mode === 'stream') {
        const response = await fetch(STREAM_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: input })
        });

        if (!response.ok) throw new Error('Stream failed');

        setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantResponse = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;
                    assistantResponse += data;
                    
                    setMessages(prev => {
                        const newMsgs = [...prev];
                        newMsgs[newMsgs.length - 1].content = assistantResponse;
                        return newMsgs;
                    });
                }
            }
        }

      } else {
        const result = await inferAdaptive(input);
        const assistantMsg = { 
            role: 'assistant', 
            content: result.response || result.message,
            meta: {
                mode: result.mode,
                domain: result.domain,
                context_used: result.context_used
            }
        };
        setMessages(prev => [...prev, assistantMsg]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'error', content: `Error: ${err.message}` }]);
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
                <div className="meta-badge">
                  {msg.meta.mode.charAt(0).toUpperCase() + msg.meta.mode.slice(1)} 
                  {msg.meta.context_used && (
                      <span className="icon" title="RAG Context Used">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginLeft: '4px'}}>
                            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                        </svg>
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
                          onClick={() => submitFeedback(index, 1)} 
                          className={`action-btn ${msg.feedback === 'up' ? 'active' : ''}`}
                          data-tooltip="Helpful"
                          disabled={!!msg.feedback} /* Strict Guardrail: One vote per message */
                          style={!!msg.feedback ? { cursor: 'not-allowed', opacity: msg.feedback === 'up' ? 1 : 0.3 } : {}}
                      >
                          {/* Thumbs Up */}
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                          </svg>
                      </button>
                      <button 
                          onClick={() => submitFeedback(index, -1)} 
                          className={`action-btn ${msg.feedback === 'down' ? 'active' : ''}`}
                          data-tooltip="Not Helpful"
                          disabled={!!msg.feedback} /* Strict Guardrail: One vote per message */
                          style={!!msg.feedback ? { cursor: 'not-allowed', opacity: msg.feedback === 'down' ? 1 : 0.3 } : {}}
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
        {isLoading && mode !== 'stream' && (
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
        <select 
            className="mode-select" 
            value={mode} 
            onChange={(e) => setMode(e.target.value)}
            title="Select Inference Mode"
        >
            <option value="adaptive" title="Uses RAG & Tools for complex queries">Adaptive Routing</option>
            <option value="stream" title="Fast, token-by-token generation">Real-time Stream</option>
        </select>
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
