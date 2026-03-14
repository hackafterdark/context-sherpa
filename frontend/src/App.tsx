import { useState, useEffect } from 'react';
import Layout from './Layout';
import Home from './Home';
import Settings from './Settings';
import Atlas from './Atlas';
import Help from './Help';
import Skills from './Skills';
import { GetPreferences, SavePreferences, GetWorkspaces } from '../wailsjs/go/main/App';

function App() {
    const [activeTab, setActiveTab] = useState('home');
    const [atlasWorkspace, setAtlasWorkspace] = useState('');
    const [theme, setTheme] = useState('dracula');
    const [loaded, setLoaded] = useState(false);

    const openAtlas = (root: string) => {
        setAtlasWorkspace(root);
        setActiveTab('atlas');
    };

    useEffect(() => {
        // Load initial preferences from Go backend
        GetPreferences().then((prefs) => {
            if (prefs && prefs.theme) {
                setTheme(prefs.theme);
            }
            setLoaded(true);
        });

        // If no atlasWorkspace is set, try to default to the first available one
        if (!atlasWorkspace) {
            GetWorkspaces().then((ws: any[]) => {
                if (ws && ws.length > 0) {
                    setAtlasWorkspace(ws[0].root);
                }
            });
        }
    }, [atlasWorkspace]);

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        // Only persist if we've successfully loaded initial state
        if (loaded) {
            SavePreferences({ theme });
        }
    }, [theme, loaded]);

    return (
        <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
            {activeTab === 'home' && <Home onVisualize={openAtlas} />}
            {activeTab === 'atlas' && <Atlas workspaceRoot={atlasWorkspace} onWorkspaceChange={setAtlasWorkspace} />}
            {activeTab === 'skills' && <Skills workspaceRoot={atlasWorkspace} onWorkspaceChange={setAtlasWorkspace} theme={theme} />}
            {activeTab === 'help' && <Help />}
            {activeTab === 'settings' && <Settings theme={theme} setTheme={setTheme} />}
        </Layout>
    );
}

export default App;
