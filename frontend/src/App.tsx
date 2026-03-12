import { useState, useEffect } from 'react';
import Layout from './Layout';
import Home from './Home';
import Settings from './Settings';
import Atlas from './Atlas';
import { GetPreferences, SavePreferences } from '../wailsjs/go/main/App';

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
    }, []);

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
            {activeTab === 'settings' && <Settings theme={theme} setTheme={setTheme} />}
        </Layout>
    );
}

export default App;
