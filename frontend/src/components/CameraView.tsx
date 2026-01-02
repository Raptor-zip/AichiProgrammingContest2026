import React, { useState, useEffect } from 'react';

// You might usually put this in a config or context
const API_BASE = "http://127.0.0.1:8000/api";

export const CameraView: React.FC = () => {
    const [streamUrl, setStreamUrl] = useState(`${API_BASE}/stream`);
    const [isCapturing, setIsCapturing] = useState(false);
    const [lastCapture, setLastCapture] = useState<string | null>(null);
    const [errorCount, setErrorCount] = useState(0);

    // To force re-render/re-connect stream if needed
    const refreshStream = () => {
        setStreamUrl(`${API_BASE}/stream?t=${Date.now()}`);
        setErrorCount(0);
    };

    const handleStreamError = () => {
        console.warn("Stream connection failed, retrying...");
        // Simple retry backoff or just show error after some tries
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
            alert("Capture successful: " + data.filename);
        } catch (error) {
            console.error(error);
            alert("Capture failed");
        } finally {
            setIsCapturing(false);
        }
    };

    return (
        <div className="flex flex-col h-full p-4 bg-dark-bg text-white">
            <div className="flex-1 relative bg-black rounded-lg overflow-hidden border-2 border-primary shadow-lg mb-4">
                <img
                    src={streamUrl}
                    alt="Camera Stream"
                    className="w-full h-full object-contain"
                    onError={handleStreamError}
                />
                {errorCount >= 5 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-red-500 font-bold">
                        Stream Offline. Check backend.
                    </div>
                )}
            </div>

            <div className="flex justify-center space-x-4">
                <button
                    onClick={refreshStream}
                    className="bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-6 rounded shadow transition"
                >
                    Reconnect Stream
                </button>
                <button
                    onClick={handleCapture}
                    disabled={isCapturing}
                    className={`bg-secondary hover:bg-secondary-hover text-white font-bold py-3 px-8 rounded-full shadow-lg transform transition hover:scale-105 ${isCapturing ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    {isCapturing ? 'Capturing...' : '📸 Capture'}
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
