import { useEffect, useState } from 'react';
import { InstallAstGrep, GetAstGrepStatus, InstallScipIndexer, GetScipIndexerStatus, OpenBinDir } from '../wailsjs/go/main/App';
import { Icon } from '@iconify/react';

type SettingsProps = {
    theme: string;
    setTheme: (theme: string) => void;
};

export default function Settings({ theme, setTheme }: SettingsProps) {
    const [installStatus, setInstallStatus] = useState<string>('');
    const [isInstalling, setIsInstalling] = useState(false);
    const [astGrepInfo, setAstGrepInfo] = useState<{ installed: boolean; version: string; path: string } | null>(null);
    const [scipGoInfo, setScipGoInfo] = useState<{ installed: boolean; version: string; path: string } | null>(null);
    const [scipTsInfo, setScipTsInfo] = useState<{ installed: boolean; version: string; path: string } | null>(null);
    const [scipPyInfo, setScipPyInfo] = useState<{ installed: boolean; version: string; path: string } | null>(null);
    const [scipStatus, setScipStatus] = useState<string>('');
    const [scipTsStatus, setScipTsStatus] = useState<string>('');
    const [scipPyStatus, setScipPyStatus] = useState<string>('');
    const [isScipInstalling, setIsScipInstalling] = useState(false);
    const [isScipTsInstalling, setIsScipTsInstalling] = useState(false);
    const [isScipPyInstalling, setIsScipPyInstalling] = useState(false);

    const loadStatus = async () => {
        try {
            const status = await GetAstGrepStatus();
            setAstGrepInfo(status as any);

            const goStatus = await GetScipIndexerStatus('go');
            setScipGoInfo(goStatus as any);

            const tsStatus = await GetScipIndexerStatus('typescript');
            setScipTsInfo(tsStatus as any);

            const pyStatus = await GetScipIndexerStatus('python');
            setScipPyInfo(pyStatus as any);
        } catch (e) {
            console.error("Error fetching tool status:", e);
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
            await loadStatus();
        } catch (e: any) {
            setInstallStatus('Error: ' + e);
        } finally {
            setIsInstalling(false);
        }
    };

    const handleScipInstall = async (lang: 'go' | 'typescript' | 'python') => {
        if (lang === 'go') {
            setIsScipInstalling(true);
            setScipStatus('Starting download...');
        } else if (lang === 'typescript') {
            setIsScipTsInstalling(true);
            setScipTsStatus('Installing via npm...');
        } else {
            setIsScipPyInstalling(true);
            setScipPyStatus('Installing via npm...');
        }

        try {
            const result = await InstallScipIndexer(lang);
            if (lang === 'go') setScipStatus(result);
            else if (lang === 'typescript') setScipTsStatus(result);
            else setScipPyStatus(result);
            await loadStatus();
        } catch (e: any) {
            if (lang === 'go') setScipStatus('Error: ' + e);
            else if (lang === 'typescript') setScipTsStatus('Error: ' + e);
            else setScipPyStatus('Error: ' + e);
        } finally {
            if (lang === 'go') setIsScipInstalling(false);
            else if (lang === 'typescript') setIsScipTsInstalling(false);
            else setIsScipPyInstalling(false);
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
                <div className="card-body gap-6">
                    <div className="flex items-center justify-between">
                        <h2 className="card-title text-xl flex items-center gap-2">
                            <Icon icon="lucide:wrench" className="text-primary" />
                            Toolchain Management
                        </h2>
                        <button
                            className="btn btn-ghost btn-sm gap-2 text-base-content/60 hover:text-base-content"
                            onClick={() => OpenBinDir()}
                            title="Open the managed binaries directory in file explorer"
                        >
                            <Icon icon="lucide:folder-open" />
                            Browse Binaries
                        </button>
                    </div>

                    <p className="text-base-content/70 text-sm -mt-4">
                        Context-Sherpa manages a dedicated toolchain in your `~/.context-sherpa/bin` directory to enable advanced code intelligence features.
                    </p>

                    {/* Structural Analysis (ast-grep) */}
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center gap-2">
                            <Icon icon="lucide:layout-template" className="text-primary w-5 h-5" />
                            <h3 className="font-bold text-lg">Structural Analysis (ast-grep)</h3>
                        </div>
                        <p className="text-base-content/60 text-sm">
                            Structural code scanning to enforce correctness, security vulnerabilities, and coding standards.
                        </p>

                        {astGrepInfo?.installed && (
                            <div className="alert alert-success bg-success/10 text-success border-success/20 py-2 flex gap-3">
                                <Icon icon="lucide:check-circle-2" className="w-4 h-4 shrink-0" />
                                <div className="flex flex-col">
                                    <span className="font-semibold text-xs">ast-grep is installed and ready.</span>
                                    <div className="flex gap-4 mt-0.5 opacity-80 font-mono text-[10px]">
                                        <span>{astGrepInfo.version || 'v0.0.0'}</span>
                                        <span className="truncate max-w-xs">{astGrepInfo.path}</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="flex items-center gap-4">
                            <button
                                className={`btn btn-sm btn-primary ${isInstalling ? 'btn-disabled' : ''}`}
                                onClick={handleInstall}
                            >
                                {isInstalling ? (
                                    <>
                                        <span className="loading loading-spinner loading-xs"></span>
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
                                <div className={`text-xs ${installStatus.startsWith('Error') ? 'text-error' : 'text-success'}`}>
                                    {installStatus}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="divider my-0 opacity-20"></div>

                    {/* Symbolic Intelligence (SCIP) */}
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center gap-2">
                            <Icon icon="lucide:network" className="text-secondary w-5 h-5" />
                            <h3 className="font-bold text-lg">Symbolic Intelligence (SCIP)</h3>
                        </div>
                        <p className="text-base-content/60 text-sm">
                            Enables cross-file navigation and relational mapping through language-specific indexers.
                        </p>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-2">
                            {[
                                { id: 'go', label: 'Go (Native)', info: scipGoInfo, status: scipStatus, isInstalling: isScipInstalling },
                                { id: 'typescript', label: 'TypeScript (NPM)', info: scipTsInfo, status: scipTsStatus, isInstalling: isScipTsInstalling },
                                { id: 'python', label: 'Python (NPM)', info: scipPyInfo, status: scipPyStatus, isInstalling: isScipPyInstalling }
                            ].map((lang) => (
                                <div key={lang.id} className="border border-base-200 rounded-lg p-4 bg-base-200/30 flex flex-col gap-3">
                                    <div className="flex items-center justify-between">
                                        <span className="font-semibold text-sm capitalize">{lang.id}</span>
                                        {lang.info?.installed ? (
                                            <div className="badge badge-success badge-sm gap-1 py-2">
                                                <Icon icon="lucide:check" className="w-3 h-3" />
                                                Ready
                                            </div>
                                        ) : (
                                            <div className="badge badge-ghost badge-sm py-2 opacity-50">Not Installed</div>
                                        )}
                                    </div>

                                    {lang.info?.installed && (
                                        <div className="flex flex-col gap-1 opacity-70">
                                            <span className="text-[10px] font-mono truncate">{lang.info.version || 'v0.0.0'}</span>
                                            <span className="text-[10px] font-mono truncate opacity-60">{lang.info.path}</span>
                                        </div>
                                    )}

                                    <div className="mt-auto pt-2 border-t border-base-200/50 flex flex-col gap-2">
                                        <button
                                            className={`btn btn-xs ${lang.info?.installed ? 'btn-ghost' : 'btn-outline btn-secondary'} ${lang.isInstalling ? 'btn-disabled' : ''}`}
                                            onClick={() => handleScipInstall(lang.id as any)}
                                        >
                                            {lang.isInstalling ? (
                                                <>
                                                    <span className="loading loading-spinner loading-xs"></span>
                                                    Installing...
                                                </>
                                            ) : (
                                                <>
                                                    <Icon icon="lucide:download" className="w-3 h-3" />
                                                    {lang.info?.installed ? 'Update' : `Install ${lang.label}`}
                                                </>
                                            )}
                                        </button>
                                        {lang.status && (
                                            <div className={`text-[10px] ${lang.status.startsWith('Error') ? 'text-error' : 'text-success'} truncate`} title={lang.status}>
                                                {lang.status}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
