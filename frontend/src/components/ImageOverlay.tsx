import React, { useState, useEffect, useRef } from 'react';

interface Box {
    x: number;
    y: number;
    w: number;
    h: number;
}

interface OCRBlock {
    box: [number, number, number, number]; // [x_min, y_min, x_max, y_max] or similar? YomiToku usually returns xyxy or xywh
    text: string;
    confidence?: number;
}

interface ImageOverlayProps {
    src: string;
    ocrData: any; // Flexible for now
    onTextClick?: (text: string) => void;
}

export const ImageOverlay: React.FC<ImageOverlayProps> = ({ src, ocrData, onTextClick }) => {
    const [imgSize, setImgSize] = useState<{ w: number, h: number } | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const { naturalWidth, naturalHeight } = e.currentTarget;
        setImgSize({ w: naturalWidth, h: naturalHeight });
    };

    // Helper to parse blocks. YomiToku structure assumption:
    // results is usually a list of lists or dicts.
    // Let's assume standard YomiToku output: an array of [bbox, text, confidence] or similar?
    // Or maybe it's a dict with "blocks".
    // We will try to handle a common format: { blocks: [...] } or [...]

    // YomiToku (SimpleOCR) behavior seen in other projects:
    // returns a list of detections: [[x_min, y_min, x_max, y_max], "text", confidence]
    // OR
    // returns something else.
    // Since we don't have the library docs here, we'll try to support a generic list format.

    const renderOverlays = () => {
        if (!imgSize || !ocrData) return null;

        // Normalize data to list of blocks
        let blocks: any[] = [];
        if (Array.isArray(ocrData)) {
            blocks = ocrData;
        } else if (ocrData.blocks && Array.isArray(ocrData.blocks)) {
            blocks = ocrData.blocks;
        } else if (ocrData.words && Array.isArray(ocrData.words)) {
            // Yomitoku often calls it "words"
            blocks = ocrData.words;
        } else if (ocrData.results && Array.isArray(ocrData.results)) {
            // Generic "results" key
            blocks = ocrData.results;
        }

        return blocks.map((block, idx) => {
            let x = 0, y = 0, w = 0, h = 0;
            let text = "";

            // Heuristic to detect format

            // Format 1: Yomitoku / generic polygon "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if (block.points && Array.isArray(block.points)) {
                const xs = block.points.map((p: any) => p[0]);
                const ys = block.points.map((p: any) => p[1]);
                const xMin = Math.min(...xs);
                const xMax = Math.max(...xs);
                const yMin = Math.min(...ys);
                const yMax = Math.max(...ys);

                x = xMin;
                y = yMin;
                w = xMax - xMin;
                h = yMax - yMin;
                text = block.content || block.text || "";
            }
            // Format 2: [ [x1, y1, x2, y2], "text", conf ] (SimpleOCR sometimes)
            else if (Array.isArray(block) && block.length >= 2 && Array.isArray(block[0])) {
                const coords = block[0]; // [x1, y1, x2, y2]
                text = block[1];
                x = coords[0];
                y = coords[1];
                w = coords[2] - coords[0];
                h = coords[3] - coords[1];
            }
            // Format 3: { box: [x,y,w,h], text: "" }
            else if (block.box) {
                x = block.box[0];
                y = block.box[1];
                w = block.box[2];
                h = block.box[3];
                text = block.text || block.content || "";
            }

            if (w === 0 || h === 0) return null;

            // Calculate percentage
            const left = (x / imgSize.w) * 100;
            const top = (y / imgSize.h) * 100;
            const width = (w / imgSize.w) * 100;
            const height = (h / imgSize.h) * 100;

            return (
                <div
                    key={idx}
                    className="absolute border border-red-500 bg-red-500 bg-opacity-10 hover:bg-opacity-30 cursor-pointer group z-10"
                    style={{
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${width}%`,
                        height: `${height}%`
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                        onTextClick && onTextClick(text);
                    }}
                    title={text}
                >
                    <span className="absolute bottom-full left-0 bg-black bg-opacity-75 text-white text-xs px-1 rounded opacity-0 group-hover:opacity-100 whitespace-nowrap pointer-events-none z-20">
                        {text}
                    </span>
                </div>
            );
        });
    };

    return (
        <div className="relative w-full h-full" ref={containerRef}>
            <img
                src={src}
                alt="Analyzed"
                className="w-full h-auto block" // Removed object-contain, using natural layout with width 100% to match overlay
                onLoad={onImgLoad}
            />
            {renderOverlays()}
        </div>
    );
};
