import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, RefreshCw } from 'lucide-react';
import { CaptureHud } from './CaptureHud';

// You might usually put this in a config or context
const API_BASE = "http://127.0.0.1:8000/api";

export const CameraView: React.FC = () => {
    const [streamKey, setStreamKey] = useState(Date.now());
    const [isCapturing, setIsCapturing] = useState(false);
    const [lastCapture, setLastCapture] = useState<string | null>(null);
    const [errorCount, setErrorCount] = useState(0);
    const [isStreamLoading, setIsStreamLoading] = useState(true);

    // Capture Status State
    const [captureProgress, setCaptureProgress] = useState(0);
    const [captureTriggered, setCaptureTriggered] = useState(false);

    // SSE ref for cleanup
    const eventSourceRef = useRef<EventSource | null>(null);
    const [sseKey, setSseKey] = useState(Date.now());

    // SSE connection with reconnect support
    useEffect(() => {
        const connectSSE = () => {
            // Close existing connection
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }

            const eventSource = new EventSource(`${API_BASE}/capture_status`);
            eventSourceRef.current = eventSource;

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setCaptureProgress(data.progress || 0);
                    setCaptureTriggered(!!data.triggered);
                } catch (e) {
                    // ignore parse error
                }
            };

            eventSource.onerror = () => {
                console.warn("SSE connection error, will auto-reconnect...");
                eventSource.close();
                // Auto-reconnect after delay
                setTimeout(connectSSE, 2000);
            };
        };

        connectSSE();

        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, [sseKey]);

    // Force complete reconnection of stream
    const refreshStream = useCallback(() => {
        setIsStreamLoading(true);
        setErrorCount(0);
        // Change key to force React to completely remount the img element
        setStreamKey(Date.now());
    }, []);

    // Also reconnect SSE when refreshing
    const handleFullReconnect = useCallback(() => {
        refreshStream();
        setSseKey(Date.now());
    }, [refreshStream]);

    const handleStreamLoad = () => {
        setIsStreamLoading(false);
        setErrorCount(0);
    };

    const handleStreamError = () => {
        console.warn("Stream connection failed, retrying...");
        setIsStreamLoading(false);
        if (errorCount < 5) {
            setErrorCount(prev => prev + 1);
            setTimeout(refreshStream, 2000);
        }
    };


    const handleCapture = async () => {
        setIsCapturing(true);
        try {
            const response = await fetch(`${API_BASE}/capture`, { method: 'POST' });
            if (!response.ok) throw new Error('Capture failed');
            const data = await response.json();
            setLastCapture(data.url);
            // Show toast or notification here
            alert("撮影成功: " + data.filename);
        } catch (error) {
            console.error(error);
            alert("撮影に失敗しました");
        } finally {
            setIsCapturing(false);
        }
    };

    return (
        <div className="flex flex-col h-full p-4 bg-dark-bg text-white">
            <div className="flex-1 relative bg-black rounded-lg overflow-hidden border-2 border-primary shadow-lg mb-4">
                {/* Loading indicator */}
                {isStreamLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
                        <div className="text-primary animate-pulse">接続中...</div>
                    </div>
                )}

                {/* Key forces complete remount of img element */}
                <img
                    key={streamKey}
                    src={`${API_BASE}/stream?t=${streamKey}`}
                    alt="Camera Stream"
                    className="w-full h-full object-contain"
                    onLoad={handleStreamLoad}
                    onError={handleStreamError}
                />

                {/* HUD Overlay */}
                <CaptureHud progress={captureProgress} triggered={captureTriggered} />

                {errorCount >= 5 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-red-500 font-bold">
                        ストリームがオフラインです。バックエンドを確認してください。
                    </div>
                )}
            </div>

            <div className="flex justify-center space-x-4">
                <button
                    onClick={handleFullReconnect}
                    className="bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-6 rounded shadow transition"
                >
                    再接続
                </button>
                <button
                    onClick={handleCapture}
                    disabled={isCapturing}
                    className={`bg-secondary hover:bg-secondary-hover text-white font-bold py-3 px-8 rounded-full shadow-lg transform transition hover:scale-105 ${isCapturing ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    {isCapturing ? '撮影中...' : '📸 撮影'}
                </button>
            </div>

            {lastCapture && (
                <div className="absolute bottom-20 right-4 w-32 h-24 bg-dark-surface border border-primary rounded overflow-hidden shadow-xl animate-bounce">
                    <img src={`http://127.0.0.1:8000${lastCapture}`} alt="Last capture" className="w-full h-full object-cover" />
                </div>
            )}
        </div>
    );
};
