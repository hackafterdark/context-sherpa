import { useState, useEffect } from 'react';
import Layout from './Layout';
import Home from './Home';
import Settings from './Settings';

function App() {
    const [activeTab, setActiveTab] = useState('home');
    const [theme, setTheme] = useState(localStorage.getItem('context-sherpa-theme') || 'dracula');

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('context-sherpa-theme', theme);
    }, [theme]);

    return (
        <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
            {activeTab === 'home' && <Home />}
            {activeTab === 'settings' && <Settings theme={theme} setTheme={setTheme} />}
        </Layout>
    );
}

export default App;
