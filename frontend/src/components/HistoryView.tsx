import React, { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { ImageOverlay } from './ImageOverlay';
import { TextActionMenu } from './TextActionMenu';

const API_BASE = "http://127.0.0.1:8000/api";

interface Capture {
    filename: string;
    filepath: string;
    created_at: string;
    url: string;
    subject?: string;
    detected_id?: number;
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

    // LLM Support State
    const [menuPos, setMenuPos] = useState<{ x: number, y: number } | null>(null);
    const [selectedText, setSelectedText] = useState<string>("");
    const [llmContent, setLlmContent] = useState<{ type: string, content: string } | null>(null);
    const [llmLoading, setLlmLoading] = useState(false);

    // Register ID State
    const [registerSubjectName, setRegisterSubjectName] = useState("");
    const [registering, setRegistering] = useState(false);

    // Auto-fetch OCR when selection changes
    useEffect(() => {
        if (selectedCapture) {
            setOcrResult(null); // Clear previous
            setLlmContent(null); // Clear LLM content
            setMenuPos(null);
            setShowOriginal(false); // Reset to processed by default
            fetchOcrResult(selectedCapture.relative_path || selectedCapture.filename);
        }
    }, [selectedCapture]);

    const handleTextClick = (text: string, e: React.MouseEvent) => {
        setSelectedText(text);
        setMenuPos({ x: e.clientX, y: e.clientY });
    };

    const handleLlmAction = async (type: 'explain' | 'problem') => {
        setMenuPos(null); // Close menu
        setLlmLoading(true);
        setLlmContent(null); // Clear previous content? or keep? Let's clear to show loading

        try {
            const res = await fetch(`${API_BASE}/study_support`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: selectedText, type })
            });
            const data = await res.json();
            setLlmContent({ type, content: data.content });
        } catch (e) {
            console.error("LLM Error", e);
            alert("Failed to get response from AI");
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
            alert("OCR Failed");
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
                alert("Registration Successful! Future captures will be sorted.");
                // Ideally update current file subject locally or reload
                fetchHistory();
            } else {
                alert("Registration failed");
            }
        } catch (e) {
            console.error(e);
            alert("Error registering ID");
        } finally {
            setRegistering(false);
        }
    };

    return (
        <div className="flex h-full bg-dark-bg text-gray-200">
            {/* List Sidebar */}
            <div className="w-1/3 border-r border-primary bg-dark-surface flex flex-col">
                <div className="p-4 border-b border-primary">
                    <h2 className="text-xl font-bold">History</h2>
                    <button onClick={fetchHistory} className="text-sm text-blue-400 hover:text-blue-300">Refresh</button>
                </div>
                <div className="flex-1 overflow-y-auto">
                    {loading ? <div className="p-4">Loading...</div> : (
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
                                    <span className="font-bold text-yellow-200">Unknown ArUco ID detected: {selectedCapture.detected_id}</span>
                                    <p className="text-xs text-yellow-300">Register this ID to a subject to auto-sort future captures.</p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="text"
                                        className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white"
                                        placeholder="Subject Name"
                                        value={registerSubjectName}
                                        onChange={(e) => setRegisterSubjectName(e.target.value)}
                                    />
                                    <button
                                        onClick={handleRegisterId}
                                        className="bg-primary hover:bg-primary-hover px-3 py-1 rounded text-sm text-white disabled:opacity-50"
                                        disabled={registering}
                                    >
                                        {registering ? "..." : "Register"}
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
                                        Processed
                                    </button>
                                    <button
                                        onClick={() => setShowOriginal(true)}
                                        className={`px-3 py-1 text-xs font-bold rounded-full transition-all duration-300 ${showOriginal ? 'bg-secondary text-white shadow-[0_0_10px_rgba(255,0,255,0.5)]' : 'text-gray-400 hover:text-white'}`}
                                    >
                                        Original
                                    </button>
                                </div>
                            )}

                            {/* Use container with max-width/height logic if needed, but ImageOverlay expects full width */}
                            <div className="w-full max-w-4xl mx-auto relative">
                                <ImageOverlay
                                    src={showOriginal && selectedCapture.url_original ? `http://127.0.0.1:8000${selectedCapture.url_original}` : `http://127.0.0.1:8000${selectedCapture.url}`}
                                    ocrData={!showOriginal ? ocrResult : null} // Only show OCR overlay on processed image (assuming OCR ran on processed)
                                    onTextClickTrigger={(text, e) => handleTextClick(text, e)}
                                />
                            </div>
                        </div>

                        {/* LLM Result Panel if active */}
                        {(llmContent || llmLoading) && (
                            <div className="bg-dark-surface rounded border border-secondary p-4 shrink-0 transition-all">
                                <div className="flex justify-between items-center mb-2 border-b border-gray-700 pb-2">
                                    <h3 className="font-bold text-secondary-light">
                                        {llmLoading ? "AI Thinking..." : (llmContent?.type === 'explain' ? '📖 Explanation' : '📝 Practice Problems')}
                                    </h3>
                                    <button onClick={() => setLlmContent(null)} className="text-gray-500 hover:text-white">✕</button>
                                </div>
                                <div className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                                    {llmLoading ? (
                                        <div className="flex items-center space-x-2 animate-pulse">
                                            <div className="w-2 h-2 bg-secondary rounded-full"></div>
                                            <div className="w-2 h-2 bg-secondary rounded-full delay-75"></div>
                                            <div className="w-2 h-2 bg-secondary rounded-full delay-150"></div>
                                        </div>
                                    ) : (
                                        <div className="prose prose-invert prose-sm max-w-none">
                                            <ReactMarkdown
                                                components={{
                                                    strong: ({ node, ...props }) => <span className="font-bold text-secondary-light" {...props} />,
                                                    h1: ({ node, ...props }) => <h1 className="text-xl font-bold my-2" {...props} />,
                                                    h2: ({ node, ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                                                    h3: ({ node, ...props }) => <h3 className="text-md font-bold my-1" {...props} />,
                                                    ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2" {...props} />,
                                                    ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2" {...props} />,
                                                    li: ({ node, ...props }) => <li className="my-1" {...props} />,
                                                    code: ({ node, ...props }) => <code className="bg-gray-800 rounded px-1" {...props} />,
                                                }}
                                            >
                                                {llmContent?.content || ""}
                                            </ReactMarkdown>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        <div className="bg-dark-surface rounded border border-gray-700 p-4 shrink-0">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="font-bold">Analysis Results</h3>
                                <div className="space-x-2">
                                    <button
                                        onClick={() => setShowRawJson(!showRawJson)}
                                        className="text-xs text-blue-400 hover:text-blue-300 underline"
                                    >
                                        {showRawJson ? "Hide JSON" : "Show JSON"}
                                    </button>
                                    <button
                                        onClick={() => runOcr(selectedCapture)}
                                        disabled={processing}
                                        className="bg-secondary hover:bg-secondary-hover px-4 py-1 rounded text-sm text-white disabled:opacity-50"
                                    >
                                        {processing ? 'Processing...' : 'Re-Run Analysis'}
                                    </button>
                                </div>
                            </div>
                            {showRawJson && (
                                <pre className="text-xs text-gray-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                                    {ocrResult ? JSON.stringify(ocrResult, null, 2) : "No analysis data found."}
                                </pre>
                            )}
                            <div className="text-sm text-gray-300">
                                {ocrResult ? "Click on the red boxes in the image to extract text." : "No text detected or analysis pending."}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                        Select an image to view details
                    </div>
                )}
            </div>
            {/* Popup Menu */}
            {
                menuPos && (
                    <TextActionMenu
                        x={menuPos.x}
                        y={menuPos.y}
                        onExplain={() => handleLlmAction('explain')}
                        onPractice={() => handleLlmAction('problem')}
                        onClose={() => setMenuPos(null)}
                    />
                )
            }
        </div >
    );
};
