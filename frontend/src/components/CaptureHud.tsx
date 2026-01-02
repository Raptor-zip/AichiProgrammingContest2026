import React from 'react';

interface CaptureHudProps {
    progress: number;
    triggered: boolean;
}

export const CaptureHud: React.FC<CaptureHudProps> = ({ progress, triggered }) => {
    if (progress <= 0 && !triggered) return null;

    // Radius for circular progress
    const r = 40;
    const c = 2 * Math.PI * r;
    const offset = c - (progress * c);

    return (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
            {/* Overlay Flash on Trigger */}
            {triggered && (
                <div className="absolute inset-0 bg-white animate-flash opacity-0"></div>
            )}

            {/* Central HUD */}
            <div className={`relative flex items-center justify-center transition-transform ${triggered ? 'scale-110' : 'scale-100'}`}>
                {/* Outer Ring Animation (Spinning) */}
                <div className="absolute w-32 h-32 border-2 border-primary/30 rounded-full animate-spin-slow"></div>
                <div className="absolute w-28 h-28 border border-secondary/20 rounded-full animate-reverse-spin"></div>

                {/* Progress Circle SVG */}
                <svg className="w-24 h-24 transform -rotate-90 drop-shadow-[0_0_10px_rgba(0,255,255,0.7)]">
                    <circle
                        cx="48"
                        cy="48"
                        r={r}
                        fill="transparent"
                        stroke="#1a1a1a"
                        strokeWidth="8"
                    />
                    <circle
                        cx="48"
                        cy="48"
                        r={r}
                        fill="transparent"
                        stroke={triggered ? "#00ff00" : "#00ffff"}
                        strokeWidth="8"
                        strokeDasharray={c}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        className="transition-all duration-100 ease-linear"
                        style={{ filter: triggered ? 'drop-shadow(0 0 5px #00ff00)' : 'drop-shadow(0 0 5px #00ffff)' }}
                    />
                </svg>

                {/* Center Text */}
                <div className="absolute text-center">
                    {triggered ? (
                        <div className="text-green-400 font-bold text-lg animate-bounce">CAPTURED</div>
                    ) : (
                        <div className="text-secondary font-mono text-xl font-bold">
                            {(progress * 100).toFixed(0)}%
                        </div>
                    )}
                </div>
            </div>

            {/* Status Text at Bottom */}
            {!triggered && (
                <div className="absolute bottom-20 text-secondary-light font-mono text-sm tracking-widest uppercase animate-pulse">
                    Target Locked... Stabilize
                </div>
            )}
        </div>
    );
};
