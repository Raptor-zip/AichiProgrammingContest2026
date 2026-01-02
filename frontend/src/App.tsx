import { useState } from 'react';
import { CameraView } from './components/CameraView';
import { HistoryView } from './components/HistoryView';
import { SettingsDialog } from './components/SettingsDialog';

type ViewMode = 'camera' | 'history';

function App() {
  const [mode, setMode] = useState<ViewMode>('camera');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  return (
    <div className="flex flex-col h-screen bg-dark-bg text-white overflow-hidden font-sans">
      {/* Header / Nav */}
      <header className="h-14 bg-gradient-to-r from-primary to-primary-active flex items-center px-4 justify-between shadow-md z-10">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-bold tracking-wide">📷 Modern GUI</h1>
          <nav className="flex space-x-2 bg-dark-bg bg-opacity-20 rounded p-1">
            <button
              onClick={() => setMode('camera')}
              className={`px-4 py-1 rounded transition text-sm font-medium ${mode === 'camera' ? 'bg-blue-500 text-white shadow' : 'hover:bg-white/10 text-gray-300'}`}
            >
              Camera
            </button>
            <button
              onClick={() => setMode('history')}
              className={`px-4 py-1 rounded transition text-sm font-medium ${mode === 'history' ? 'bg-purple-600 text-white shadow' : 'hover:bg-white/10 text-gray-300'}`}
            >
              History / AI
            </button>
          </nav>
        </div>

        <button
          onClick={() => setIsSettingsOpen(true)}
          className="p-2 rounded-full hover:bg-white/10 transition"
          title="Settings"
        >
          ⚙️
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 relative overflow-hidden">
        {mode === 'camera' && <CameraView />}
        {mode === 'history' && <HistoryView />}
      </main>

      {/* Settings Dialog */}
      <SettingsDialog isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  );
}

export default App;
