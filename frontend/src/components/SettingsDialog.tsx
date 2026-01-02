import React, { useEffect, useState } from 'react';

const API_BASE = "http://127.0.0.1:8000/api";

interface SettingsDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

export const SettingsDialog: React.FC<SettingsDialogProps> = ({ isOpen, onClose }) => {
    const [mappings, setMappings] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            fetchSettings();
        }
    }, [isOpen]);

    const fetchSettings = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/settings`);
            const data = await res.json();
            setMappings(data.mappings || {});
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        try {
            await fetch(`${API_BASE}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mappings })
            });
            onClose();
            alert("Settings saved!");
        } catch (e) {
            alert("Failed to save settings");
        }
    };

    const updateMapping = (id: string, subject: string) => {
        setMappings(prev => ({ ...prev, [id]: subject }));
    };

    const removeMapping = (id: string) => {
        const next = { ...mappings };
        delete next[id];
        setMappings(next);
    };

    const addMapping = () => {
        // Find next available ID or just placeholder
        const nextId = Math.max(0, ...Object.keys(mappings).map(k => parseInt(k) || 0)) + 1;
        updateMapping(String(nextId), "New Subject");
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
            <div className="bg-dark-surface w-full max-w-2xl rounded-lg shadow-2xl border border-primary p-6">
                <h2 className="text-2xl font-bold mb-4 text-white">⚙️ Settings</h2>

                <div className="max-h-96 overflow-y-auto mb-4 border border-gray-700 rounded bg-dark-bg p-2">
                    <table className="w-full text-left text-sm">
                        <thead className="text-gray-400 border-b border-gray-700">
                            <tr>
                                <th className="p-2">Marker ID</th>
                                <th className="p-2">Subject Name</th>
                                <th className="p-2">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(mappings).map(([id, subject]) => (
                                <tr key={id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800 transition">
                                    <td className="p-2">
                                        <input
                                            value={id}
                                            onChange={(e) => {
                                                const newId = e.target.value;
                                                // Handle ID change logic (bit complex, maybe just readonly ID for now or delete/re-add)
                                                // Simplified: Readonly ID, use add/remove
                                            }}
                                            className="bg-transparent border border-gray-600 rounded px-2 py-1 w-20 text-center"
                                            readOnly // For simplicity
                                        />
                                    </td>
                                    <td className="p-2">
                                        <input
                                            value={subject}
                                            onChange={(e) => updateMapping(id, e.target.value)}
                                            className="bg-transparent border border-gray-600 rounded px-2 py-1 w-full"
                                        />
                                    </td>
                                    <td className="p-2">
                                        <button
                                            onClick={() => removeMapping(id)}
                                            className="text-red-500 hover:text-red-400"
                                        >
                                            🗑️
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {Object.keys(mappings).length === 0 && <div className="p-4 text-center text-gray-500">No mappings set.</div>}
                </div>

                <div className="flex justify-between items-center">
                    <button
                        onClick={addMapping}
                        className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded text-sm"
                    >
                        ➕ Add Row
                    </button>

                    <div className="space-x-2">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 rounded hover:bg-gray-700 text-gray-300"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="bg-primary hover:bg-primary-hover text-white px-6 py-2 rounded font-bold shadow"
                        >
                            💾 Save
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
