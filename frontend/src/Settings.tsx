import { useEffect, useState } from 'react';
import { InstallAstGrep, GetAstGrepStatus } from '../wailsjs/go/main/App';
import { Icon } from '@iconify/react';

type SettingsProps = {
    theme: string;
    setTheme: (theme: string) => void;
};

export default function Settings({ theme, setTheme }: SettingsProps) {
    const [installStatus, setInstallStatus] = useState<string>('');
    const [isInstalling, setIsInstalling] = useState(false);
    const [astGrepInfo, setAstGrepInfo] = useState<{ installed: boolean; version: string; path: string } | null>(null);

    const loadStatus = async () => {
        try {
            const status = await GetAstGrepStatus();
            setAstGrepInfo(status as any);
        } catch (e) {
            console.error("Error fetching ast-grep status:", e);
        }
    };

    useEffect(() => {
        loadStatus();
    }, []);

    const handleInstall = async () => {
        setIsInstalling(true);
        setInstallStatus('Starting download...');
        try {
            const result = await InstallAstGrep();
            setInstallStatus(result);
            await loadStatus(); // Refresh status after install
        } catch (e: any) {
            setInstallStatus('Error: ' + e);
        } finally {
            setIsInstalling(false);
        }
    };

    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-300">
            <h1 className="text-3xl font-bold font-sans tracking-tight">Settings</h1>

            <div className="card bg-base-100 shadow-sm border border-base-200">
                <div className="card-body">
                    <h2 className="card-title text-xl mb-4 flex items-center gap-2">
                        <Icon icon="lucide:palette" className="text-primary" />
                        Appearance
                    </h2>
                    <div className="flex items-center justify-between">
                        <span className="text-base-content/70">Interface Theme</span>
                        <select
                            className="select select-bordered select-sm w-48"
                            value={theme}
                            onChange={(e) => setTheme(e.target.value)}
                        >
                            <option value="light">Light</option>
                            <option value="dracula">Dracula (Dark)</option>
                            <option value="dark">Default Dark</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="card bg-base-100 shadow-sm border border-base-200">
                <div className="card-body">
                    <h2 className="card-title text-xl mb-2 flex items-center gap-2">
                        <Icon icon="lucide:download-cloud" className="text-primary" />
                        Toolchain Management
                    </h2>
                    <p className="text-base-content/70 text-sm mb-4">
                        Context-Sherpa requires `ast-grep` to identify structural changes and map the symbolic graph of your codebase.
                        If you don't already have it installed globally, we can manage a dedicated version in your `~/.context-sherpa/bin` directory.
                    </p>

                    {astGrepInfo?.installed && (
                        <div className="alert alert-success bg-success/10 text-success border-success/20 mb-4 py-3 flex gap-4">
                            <Icon icon="lucide:check-circle-2" className="w-5 h-5 shrink-0" />
                            <div className="flex flex-col">
                                <span className="font-semibold text-sm">ast-grep is installed and ready.</span>
                                <span className="text-xs opacity-80 font-mono mt-1">Version: {astGrepInfo.version || 'Unknown'}</span>
                                <span className="text-xs opacity-80 font-mono">Location: {astGrepInfo.path}</span>
                            </div>
                        </div>
                    )}

                    <div className="flex flex-col gap-4">
                        <div className="flex items-center gap-4">
                            <button
                                className={`btn btn-primary ${isInstalling ? 'btn-disabled' : ''}`}
                                onClick={handleInstall}
                            >
                                {isInstalling ? (
                                    <>
                                        <span className="loading loading-spinner"></span>
                                        Installing...
                                    </>
                                ) : (
                                    <>
                                        <Icon icon="lucide:download" />
                                        Install ast-grep
                                    </>
                                )}
                            </button>

                            {installStatus && (
                                <div className={`text-sm ${installStatus.startsWith('Error') ? 'text-error' : 'text-success'}`}>
                                    {installStatus}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
