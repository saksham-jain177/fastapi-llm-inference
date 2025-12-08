
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { useNavigate } from 'react-router-dom';
import { inferAdaptive, STREAM_ENDPOINT } from '../api';
import './ChatInterface.css';

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      if (mode === 'stream') {
        // SSE Streaming Implementation
        const response = await fetch(STREAM_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: input })
        });

        if (!response.ok) throw new Error('Stream failed');

        // Create a placeholder message for the assistant
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
                    if (data === '[DONE]') break; // Standard SSE convention, though our backend might just close stream
                    assistantResponse += data;
                    
                    // Update the last message (streaming effect)
                    setMessages(prev => {
                        const newMsgs = [...prev];
                        newMsgs[newMsgs.length - 1].content = assistantResponse;
                        return newMsgs;
                    });
                }
            }
        }

      } else {
        // Adaptive Mode (Standard JSON)
        const result = await inferAdaptive(input);
        const assistantMsg = { 
            role: 'assistant', 
            content: result.response || result.message, // Fallback
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
                  {msg.meta.mode} {msg.meta.context_used ? '📚' : ''}
                </div>
              )}
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {isLoading && mode !== 'stream' && <div className="loading">Thinking...</div>}
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
