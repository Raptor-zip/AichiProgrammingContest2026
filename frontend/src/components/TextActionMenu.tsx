import React from 'react';

interface TextActionMenuProps {
    x: number;
    y: number;
    onExplain: () => void;
    onPractice: () => void;
    onClose: () => void;
}

export const TextActionMenu: React.FC<TextActionMenuProps> = ({ x, y, onExplain, onPractice, onClose }) => {
    return (
        <div
            className="fixed z-50 bg-gray-800 border border-gray-600 rounded shadow-lg p-2 flex flex-col gap-2"
            style={{ left: x, top: y }}
        >
            <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-gray-400 font-bold">アクション</span>
                <button onClick={onClose} className="text-gray-500 hover:text-white text-xs">✕</button>
            </div>
            <button
                onClick={onExplain}
                className="bg-blue-600 hover:bg-blue-500 text-white text-sm px-3 py-1.5 rounded text-left flex items-center"
            >
                <span className="mr-2">📖</span> 解説
            </button>
            <button
                onClick={onPractice}
                className="bg-green-600 hover:bg-green-500 text-white text-sm px-3 py-1.5 rounded text-left flex items-center"
            >
                <span className="mr-2">📝</span> 問題作成
            </button>
        </div>
    );
};
