import React, { useEffect, useState } from 'react';
import { ImageOverlay } from './ImageOverlay';

const API_BASE = "http://127.0.0.1:8000/api";

interface Capture {
    filename: string;
    filepath: string;
    created_at: string;
    url: string;
    subject?: string;
}

export const HistoryView: React.FC = () => {
    const [captures, setCaptures] = useState<Capture[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCapture, setSelectedCapture] = useState<Capture | null>(null);
    const [ocrResult, setOcrResult] = useState<any>(null);
    const [processing, setProcessing] = useState(false);
    const [showRawJson, setShowRawJson] = useState(false);

    // Auto-fetch OCR when selection changes
    useEffect(() => {
        if (selectedCapture) {
            setOcrResult(null); // Clear previous
            fetchOcrResult(selectedCapture.filename);
        }
    }, [selectedCapture]);

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
                        <div className="bg-black rounded border border-gray-700 flex items-center justify-center p-2 relative shrink-0">
                            {/* Use container with max-width/height logic if needed, but ImageOverlay expects full width */}
                            <div className="w-full max-w-4xl mx-auto relative">
                                <ImageOverlay
                                    src={`http://127.0.0.1:8000${selectedCapture.url}`}
                                    ocrData={ocrResult}
                                    onTextClick={(text) => alert(`Selected: ${text}`)}
                                />
                            </div>
                        </div>
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
        </div>
    );
};
