import React, { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { ImageOverlay } from './ImageOverlay';

const API_BASE = "http://127.0.0.1:8000/api";

interface Capture {
    filename: string;
    filepath: string;
    created_at: string;
    url: string;
    subject?: string;
    detected_id?: number;
    relative_path?: string;
    url_original?: string | null;
}

export const HistoryView: React.FC = () => {
    const [captures, setCaptures] = useState<Capture[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCapture, setSelectedCapture] = useState<Capture | null>(null);
    const [ocrResult, setOcrResult] = useState<any>(null);
    const [processing, setProcessing] = useState(false);
    const [showRawJson, setShowRawJson] = useState(false);
    const [showOriginal, setShowOriginal] = useState(false);

    // LLM Support State - Multiple selection
    const [selectedTexts, setSelectedTexts] = useState<Set<number>>(new Set());
    const [llmLoading, setLlmLoading] = useState(false);

    // Chat State
    const [chatHistory, setChatHistory] = useState<{ role: string, content: string }[]>([]);
    const [chatInput, setChatInput] = useState("");
    const [chatContext, setChatContext] = useState<string>("");  // Store document context for chat
    const chatEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll chat to bottom
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory, llmLoading]);

    // Register ID State
    const [registerSubjectName, setRegisterSubjectName] = useState("");
    const [registering, setRegistering] = useState(false);

    // Auto-fetch OCR when selection changes
    useEffect(() => {
        if (selectedCapture) {
            setOcrResult(null); // Clear previous
            setChatHistory([]); // Clear chat history
            setChatInput("");
            setChatContext("");
            setSelectedTexts(new Set()); // Clear selections
            setShowOriginal(false); // Reset to processed by default
            fetchOcrResult(selectedCapture.relative_path || selectedCapture.filename);
        }
    }, [selectedCapture]);

    // Toggle text selection (multiple selection support)
    const handleTextToggle = (index: number) => {
        setSelectedTexts(prev => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    // Clear all selections
    const clearSelections = () => {
        setSelectedTexts(new Set());
    };

    // Get OCR blocks as array
    const getOcrBlocks = (ocr: any): any[] => {
        if (!ocr) return [];
        if (Array.isArray(ocr)) return ocr;
        if (ocr.blocks && Array.isArray(ocr.blocks)) return ocr.blocks;
        if (ocr.words && Array.isArray(ocr.words)) return ocr.words;
        if (ocr.results && Array.isArray(ocr.results)) return ocr.results;
        return [];
    };

    // Extract text from a single block
    const getBlockText = (block: any): string => {
        if (block.points) return block.content || block.text || "";
        if (Array.isArray(block) && block.length >= 2) return block[1] || "";
        if (block.box) return block.text || block.content || "";
        return "";
    };

    // Extract all text from OCR result for context
    const extractFullText = (ocr: any): string => {
        return getOcrBlocks(ocr).map(getBlockText).filter(t => t).join("\n");
    };

    // Extract selected texts
    const getSelectedTextsString = (): string => {
        const blocks = getOcrBlocks(ocrResult);
        return Array.from(selectedTexts)
            .map(idx => getBlockText(blocks[idx]))
            .filter(t => t)
            .join("\n");
    };

    const handleLlmAction = async (type: 'explain' | 'problem') => {
        if (selectedTexts.size === 0) return;

        setLlmLoading(true);

        try {
            const context = extractFullText(ocrResult);
            const selectedText = getSelectedTextsString();

            // Store context for future chat
            setChatContext(context);

            const res = await fetch(`${API_BASE}/study_support`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: selectedText, type, context })
            });
            const data = await res.json();

            // Initialize chat with the first exchange
            const userMessage = type === 'explain'
                ? `「${selectedText.slice(0, 50)}${selectedText.length > 50 ? '...' : ''}」について解説してください`
                : `「${selectedText.slice(0, 50)}${selectedText.length > 50 ? '...' : ''}」に関する練習問題を作成してください`;

            setChatHistory([
                { role: 'user', content: userMessage },
                { role: 'assistant', content: data.content }
            ]);
            clearSelections(); // Clear after submitting
        } catch (e) {
            console.error("LLM Error", e);
            alert("AIからの応答の取得に失敗しました");
        } finally {
            setLlmLoading(false);
        }
    };

    // Send follow-up message
    const handleSendMessage = async () => {
        if (!chatInput.trim() || llmLoading) return;

        const userMessage = chatInput.trim();
        setChatInput("");
        setLlmLoading(true);

        // Add user message to history immediately
        const newHistory = [...chatHistory, { role: 'user', content: userMessage }];
        setChatHistory(newHistory);

        try {
            const res = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    history: chatHistory,
                    context: chatContext
                })
            });
            const data = await res.json();

            // Add AI response to history
            setChatHistory([...newHistory, { role: 'assistant', content: data.content }]);
        } catch (e) {
            console.error("Chat Error", e);
            // Remove the user message if failed
            setChatHistory(chatHistory);
            alert("送信に失敗しました");
        } finally {
            setLlmLoading(false);
        }
    };

    const fetchOcrResult = async (filename: string) => {
        try {
            const res = await fetch(`${API_BASE}/captures/${filename}/ocr`);
            const data = await res.json();
            if (data.status === 'ok') {
                setOcrResult(data.results);
            } else {
                // No result yet, maybe processing?
                // Optionally polling could be implemented here
            }
        } catch (e) {
            console.error("Failed to fetch OCR json", e);
        }
    };

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/history`);
            const data = await res.json();
            setCaptures(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    const runOcr = async (capture: Capture) => {
        setProcessing(true);
        try {
            const res = await fetch(`${API_BASE}/ocr`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: capture.filepath, use_last_capture: false })
            });
            const data = await res.json();
            setOcrResult(data);
        } catch (e) {
            console.error(e);
            alert("OCRに失敗しました");
        } finally {
            setProcessing(false);
        }
    };

    const handleRegisterId = async () => {
        if (!selectedCapture?.detected_id || !registerSubjectName) return;
        setRegistering(true);
        try {
            // 1. Fetch current settings
            const settingsRes = await fetch(`${API_BASE}/settings`);
            const settings = await settingsRes.json();

            // 2. Update mappings
            const newMappings = { ...settings.mappings, [selectedCapture.detected_id]: registerSubjectName };

            const updateRes = await fetch(`${API_BASE}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mappings: newMappings })
            });

            if (updateRes.ok) {
                alert("登録成功！今後の撮影は自動的に分類されます。");
                // Ideally update current file subject locally or reload
                fetchHistory();
            } else {
                alert("登録に失敗しました");
            }
        } catch (e) {
            console.error(e);
            alert("ID登録中にエラーが発生しました");
        } finally {
            setRegistering(false);
        }
    };

    return (
        <div className="flex h-full bg-dark-bg text-gray-200">
            {/* List Sidebar */}
            <div className="w-1/3 border-r border-primary bg-dark-surface flex flex-col">
                <div className="p-4 border-b border-primary">
                    <h2 className="text-xl font-bold">履歴</h2>
                    <button onClick={fetchHistory} className="text-sm text-blue-400 hover:text-blue-300">更新</button>
                </div>
                <div className="flex-1 overflow-y-auto">
                    {loading ? <div className="p-4">読み込み中...</div> : (
                        captures.map(c => (
                            <div
                                key={c.filename}
                                onClick={() => { setSelectedCapture(c); setOcrResult(null); }}
                                className={`p-4 border-b border-gray-700 cursor-pointer hover:bg-primary-active transition ${selectedCapture?.filename === c.filename ? 'bg-primary' : ''}`}
                            >
                                <div className="font-medium truncate">{c.filename}</div>
                                <div className="text-xs text-gray-400 mt-1 flex items-center">
                                    {c.subject && <span className="mr-2 px-1.5 py-0.5 bg-gray-700 rounded text-gray-300 text-[10px]">{c.subject}</span>}
                                    {new Date(c.created_at).toLocaleString()}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Detail View */}
            <div className="flex-1 bg-dark-bg overflow-y-auto">
                {selectedCapture ? (
                    <div className="flex flex-col min-h-full p-6 space-y-4">
                        {/* Unknown ID Prompt */}
                        {selectedCapture.detected_id && selectedCapture.subject === "Unclassified" && (
                            <div className="bg-yellow-900/50 border border-yellow-600 p-4 rounded flex items-center justify-between shrink-0">
                                <div>
                                    <span className="font-bold text-yellow-200">未登録のArUco IDを検出: {selectedCapture.detected_id}</span>
                                    <p className="text-xs text-yellow-300">このIDを教科に登録すると、今後の撮影が自動的に分類されます。</p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="text"
                                        className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white"
                                        placeholder="教科名"
                                        value={registerSubjectName}
                                        onChange={(e) => setRegisterSubjectName(e.target.value)}
                                    />
                                    <button
                                        onClick={handleRegisterId}
                                        className="bg-primary hover:bg-primary-hover px-3 py-1 rounded text-sm text-white disabled:opacity-50"
                                        disabled={registering}
                                    >
                                        {registering ? "..." : "登録"}
                                    </button>
                                </div>
                            </div>
                        )}



                        <div className="bg-black rounded border border-gray-700 flex flex-col items-center justify-center p-2 relative shrink-0">
                            {/* Toggle Switch */}
                            {selectedCapture.url_original && (
                                <div className="absolute top-4 right-4 z-10 flex items-center bg-dark-bg/80 backdrop-blur rounded-full p-1 border border-primary/50 shadow-lg">
                                    <button
                                        onClick={() => setShowOriginal(false)}
                                        className={`px-3 py-1 text-xs font-bold rounded-full transition-all duration-300 ${!showOriginal ? 'bg-primary text-white shadow-[0_0_10px_rgba(0,255,255,0.5)]' : 'text-gray-400 hover:text-white'}`}
                                    >
                                        補正後
                                    </button>
                                    <button
                                        onClick={() => setShowOriginal(true)}
                                        className={`px-3 py-1 text-xs font-bold rounded-full transition-all duration-300 ${showOriginal ? 'bg-secondary text-white shadow-[0_0_10px_rgba(255,0,255,0.5)]' : 'text-gray-400 hover:text-white'}`}
                                    >
                                        元画像
                                    </button>
                                </div>
                            )}

                            {/* Use container with max-width/height logic if needed, but ImageOverlay expects full width */}
                            <div className="w-full max-w-4xl mx-auto relative">
                                <ImageOverlay
                                    src={showOriginal && selectedCapture.url_original ? `http://127.0.0.1:8000${selectedCapture.url_original}` : `http://127.0.0.1:8000${selectedCapture.url}`}
                                    ocrData={!showOriginal ? ocrResult : null}
                                    selectedIndices={selectedTexts}
                                    onTextClickTrigger={(idx) => handleTextToggle(idx)}
                                />
                            </div>

                            {/* Selection Action Buttons - Fixed at bottom of screen */}
                            {selectedTexts.size > 0 && (
                                <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-dark-bg/95 backdrop-blur-md px-6 py-4 rounded-full border-2 border-primary shadow-[0_0_30px_rgba(0,255,255,0.3)] z-50">
                                    <span className="text-sm text-gray-300 font-medium">{selectedTexts.size}件選択中</span>
                                    <button
                                        onClick={() => handleLlmAction('explain')}
                                        disabled={llmLoading}
                                        className="bg-blue-600 hover:bg-blue-500 text-white text-sm px-4 py-2 rounded-full flex items-center gap-2 disabled:opacity-50 transition"
                                    >
                                        📖 解説
                                    </button>
                                    <button
                                        onClick={() => handleLlmAction('problem')}
                                        disabled={llmLoading}
                                        className="bg-green-600 hover:bg-green-500 text-white text-sm px-4 py-2 rounded-full flex items-center gap-2 disabled:opacity-50 transition"
                                    >
                                        📝 問題作成
                                    </button>
                                    <button
                                        onClick={clearSelections}
                                        className="text-gray-400 hover:text-white text-sm px-2"
                                    >
                                        ✕
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Chat Panel - Main Feature */}
                        {(chatHistory.length > 0 || llmLoading) && (
                            <div className="bg-gradient-to-br from-dark-surface to-gray-900 rounded-xl border-2 border-secondary shadow-[0_0_20px_rgba(255,0,255,0.2)] flex flex-col flex-1 min-h-[400px] max-h-[70vh] transition-all">
                                {/* Header */}
                                <div className="flex justify-between items-center p-4 border-b border-secondary/30">
                                    <h3 className="text-xl font-bold text-secondary-light flex items-center gap-2">
                                        💬 AI チャット
                                    </h3>
                                    <button onClick={() => { setChatHistory([]); setChatContext(""); }} className="text-gray-400 hover:text-white text-xl px-2">✕</button>
                                </div>

                                {/* Chat Messages */}
                                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                                    {chatHistory.map((msg, idx) => (
                                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                            <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                                                    ? 'bg-primary text-white rounded-br-sm'
                                                    : 'bg-gray-800 text-gray-100 rounded-bl-sm'
                                                }`}>
                                                {msg.role === 'user' ? (
                                                    <p className="text-sm">{msg.content}</p>
                                                ) : (
                                                    <div className="prose prose-invert prose-sm max-w-none">
                                                        <ReactMarkdown
                                                            components={{
                                                                strong: ({ node, ...props }) => <span className="font-bold text-secondary-light" {...props} />,
                                                                h1: ({ node, ...props }) => <h1 className="text-lg font-bold my-2 text-white" {...props} />,
                                                                h2: ({ node, ...props }) => <h2 className="text-base font-bold my-2 text-white" {...props} />,
                                                                h3: ({ node, ...props }) => <h3 className="text-sm font-bold my-1 text-white" {...props} />,
                                                                p: ({ node, ...props }) => <p className="my-2 leading-6 text-sm" {...props} />,
                                                                ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2 space-y-1 text-sm" {...props} />,
                                                                ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2 space-y-1 text-sm" {...props} />,
                                                                li: ({ node, ...props }) => <li className="leading-6" {...props} />,
                                                                code: ({ node, ...props }) => <code className="bg-gray-700 rounded px-1.5 py-0.5 text-secondary-light text-xs" {...props} />,
                                                            }}
                                                        >
                                                            {msg.content}
                                                        </ReactMarkdown>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                    {llmLoading && (
                                        <div className="flex justify-start">
                                            <div className="bg-gray-800 rounded-2xl rounded-bl-sm px-4 py-3">
                                                <div className="flex items-center space-x-2">
                                                    <div className="w-2 h-2 bg-secondary rounded-full animate-bounce"></div>
                                                    <div className="w-2 h-2 bg-secondary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                                    <div className="w-2 h-2 bg-secondary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                    <div ref={chatEndRef} />
                                </div>

                                {/* Input Area */}
                                <div className="p-4 border-t border-secondary/30">
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            value={chatInput}
                                            onChange={(e) => setChatInput(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                                            placeholder="質問を入力..."
                                            className="flex-1 bg-gray-800 border border-gray-600 rounded-full px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-secondary"
                                            disabled={llmLoading}
                                        />
                                        <button
                                            onClick={handleSendMessage}
                                            disabled={llmLoading || !chatInput.trim()}
                                            className="bg-secondary hover:bg-secondary-hover text-white px-6 py-2 rounded-full disabled:opacity-50 transition"
                                        >
                                            送信
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                        {/* Compact toolbar */}
                        <div className="flex items-center justify-end gap-3 text-xs shrink-0">
                            <button
                                onClick={() => setShowRawJson(!showRawJson)}
                                className="text-gray-400 hover:text-blue-400"
                            >
                                {showRawJson ? "JSONを隠す" : "JSONを表示"}
                            </button>
                            <button
                                onClick={() => runOcr(selectedCapture)}
                                disabled={processing}
                                className="text-gray-400 hover:text-secondary disabled:opacity-50"
                            >
                                {processing ? '処理中...' : '再解析'}
                            </button>
                        </div>
                        {showRawJson && (
                            <pre className="text-xs text-gray-400 bg-dark-surface rounded p-2 whitespace-pre-wrap max-h-48 overflow-y-auto shrink-0">
                                {ocrResult ? JSON.stringify(ocrResult, null, 2) : "解析データがありません。"}
                            </pre>
                        )}
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                        画像を選択して詳細を表示
                    </div>
                )}
            </div>
        </div >
    );
};
