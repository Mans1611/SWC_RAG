import { useMemo, useState } from 'react';

type ApiResponse = {
  video_id: string;
  llm_output: string;
  start: number;
  end: number;
};

type Message = {
  role: 'user' | 'assistant';
  text: string;
  videoId?: string;
  start?: number;
  end?: number;
};

type Chat = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
};

const API_URL = 'http://localhost:8000/generete/';

function App() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [activeChatId, chats],
  );

  const handleNewChat = () => {
    const id = crypto.randomUUID();
    const newChat: Chat = {
      id,
      title: 'New conversation',
      messages: [],
      createdAt: Date.now(),
    };
    setChats((previous) => [newChat, ...previous]);
    setActiveChatId(id);
    setError(null);
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    setError(null);
    setLoading(true);

    let chat = activeChat;
    if (!chat) {
      const id = crypto.randomUUID();
      chat = {
        id,
        title: trimmed.slice(0, 28),
        messages: [],
        createdAt: Date.now(),
      };
      setChats((previous) => [chat!, ...previous]);
      setActiveChatId(id);
    }

    const userMessage: Message = {
      role: 'user',
      text: trimmed,
    };

    const updatedChat: Chat = {
      ...chat,
      messages: [...chat.messages, userMessage],
      title: chat.title === 'New conversation' ? trimmed.slice(0, 28) : chat.title,
    };
    setChats((previous) => previous.map((item) => (item.id === updatedChat.id ? updatedChat : item)));
    setInput('');

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_question: trimmed }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      let data = (await response.json()) ;
      data = JSON.parse(data)
      const assistantMessage: Message = {
        role: 'assistant',
        text: data.llm_output,
        videoId: data.video_id,
        start: data.start,
        end: data.end,
      };

      const finalChat: Chat = {
        ...updatedChat,
        messages: [...updatedChat.messages, assistantMessage],
      };
      setChats((previous) => previous.map((item) => (item.id === finalChat.id ? finalChat : item)));
    } catch (err) {
      setError((err as Error).message || 'Unable to complete request.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectChat = (chatId: string) => {
    setActiveChatId(chatId);
    setError(null);
  };

  const renderMessage = (message: Message, index: number) => {
    const isUser = message.role === 'user';
    return (
      <div key={index} className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
        <div className="message-role">{isUser ? 'You' : 'Assistant'}</div>
        <div className="message-text">{message.text}</div>
        {!isUser && message.videoId && (
          <div className="video-block">
            <div className="video-wrapper">
              <iframe
                title={`video-${message.videoId}-${message.start}`}
                src={`https://www.youtube.com/embed/${message.videoId}?start=${Math.floor(
                  message.start ?? 0,
                )}&end=${Math.floor(message.end ?? 0)}&rel=0&modestbranding=1`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
            <div className="video-meta">
              <span>Start: {message.start?.toFixed(0)}s</span>
              <span>End: {message.end?.toFixed(0)}s</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <h1>YouTube RAG</h1>
            <p>Dark red AI chat</p>
          </div>
          <button className="new-chat-button" onClick={handleNewChat}>
            + New chat
          </button>
        </div>

        <div className="chat-list">
          {chats.length === 0 && <div className="empty-state">Start a conversation to save chats.</div>}
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={`chat-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => handleSelectChat(chat.id)}
            >
              <span>{chat.title}</span>
              <small>{new Date(chat.createdAt).toLocaleString()}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="main-panel">
        <div className="main-header">
          <div>
            <h2>{activeChat ? activeChat.title : 'Start a new chat'}</h2>
            <p>Send a message, then the backend will return the matching YouTube clip.</p>
          </div>
        </div>

        <div className="chat-area">
          {activeChat ? (
            activeChat.messages.map(renderMessage)
          ) : (
            <div className="empty-chat">Send a message to generate your first result.</div>
          )}
        </div>

        <div className="input-area">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask something about YouTube content..."
            rows={3}
            dir='rtl'
          />
          <div className="input-footer">
            {error && <span className="error-text">{error}</span>}
            <button className="send-button" onClick={handleSend} disabled={loading || !input.trim()}>
              {loading ? 'Sending…' : 'Send'}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
