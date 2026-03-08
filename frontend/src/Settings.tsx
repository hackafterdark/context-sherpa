import { useEffect, useState } from 'react';
import { InstallAstGrep, GetAstGrepStatus, InstallScipIndexer, GetScipIndexerStatus, OpenConfigDir, ListLocalModels, DownloadModel, GetDownloadProgress, DeleteModel, DeleteAstGrep, DeleteScipIndexer } from '../wailsjs/go/main/App';
import { Icon } from '@iconify/react';

type SettingsProps = {
    theme: string;
    setTheme: (theme: string) => void;
};

const DeleteAction = ({
    id,
    confirmDelete,
    onDelete,
    label = "Delete",
    className = "",
    size = "btn-sm"
}: {
    id: string;
    confirmDelete: string | null;
    onDelete: (id: string) => void;
    label?: string;
    className?: string;
    size?: "btn-sm" | "btn-xs";
}) => {
    const isConfirming = confirmDelete === id;
    const fixedWidth = "120px";

    return (
        <button
            className={`btn ${size} btn-ghost transition-all duration-200 justify-center flex items-center hover:bg-error/10 ${isConfirming
                ? 'text-error !opacity-100 animate-pulse'
                : 'text-error/50 hover:text-error'
                } ${className}`}
            onClick={() => onDelete(id)}
            style={{ width: fixedWidth }}
        >
            {isConfirming ? (
                'Confirm?'
            ) : (
                <>
                    <Icon icon="lucide:trash-2" className="w-4 h-4" />
                    {label && <span className="ml-1">{label}</span>}
                </>
            )}
        </button>
    );
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
    const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
    const [isScipPyInstalling, setIsScipPyInstalling] = useState(false);

    const [localModels, setLocalModels] = useState<any[]>([]);
    const [downloadingModels, setDownloadingModels] = useState<Record<string, number>>({});
    const [curatedModels] = useState([
        { id: 'smollm2-135m', name: 'SmolLM2-135M (GGUF)', type: 'Tiny', size: '145MB', url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf' },
        { id: 'qwen2.5-coder-0.5b', name: 'Qwen2.5-Coder-0.5B (GGUF)', type: 'Standard', size: '380MB', url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf' },
    ]);

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

            const models = await ListLocalModels();
            setLocalModels(models || []);


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



    const handleModelDownload = async (model: any) => {
        try {
            await DownloadModel(model.id, model.url);
            setDownloadingModels(prev => ({ ...prev, [model.id]: 0.1 })); // Start tracking
        } catch (e) {
            console.error("Failed to start download:", e);
        }
    };

    const handleToolDelete = async (type: 'ast-grep' | 'scip-go' | 'scip-typescript' | 'scip-python' | string) => {
        if (confirmDelete !== type) {
            setConfirmDelete(type);
            setTimeout(() => setConfirmDelete(null), 3000); // Reset after 3s
            return;
        }

        setConfirmDelete(null);
        try {
            if (type === 'ast-grep') {
                await DeleteAstGrep();
            } else if (type.startsWith('scip-')) {
                await DeleteScipIndexer(type.replace('scip-', ''));
            } else {
                await DeleteModel(type);
            }
            await loadStatus();
        } catch (e) {
            console.error("Failed to delete:", e);
            alert("Deletion failed: " + e);
        }
    };

    useEffect(() => {
        // Listen for completion events from backend
        // Use window.runtime or window.Wails if available
        const runtime = (window as any).runtime;
        if (!runtime) return;

        const unlistenComplete = runtime.EventsOn("model-download-complete", (id: string) => {
            setDownloadingModels(prev => {
                const next = { ...prev };
                delete next[id];
                return next;
            });
            loadStatus();
        });

        const unlistenFailed = runtime.EventsOn("model-download-failed", (data: { modelId: string, error: string }) => {
            setDownloadingModels(prev => {
                const next = { ...prev };
                delete next[data.modelId];
                return next;
            });
            alert(`Download failed for ${data.modelId}: ${data.error}`);
        });

        return () => {
            unlistenComplete();
            unlistenFailed();
        };
    }, []);

    useEffect(() => {
        const activeDownloads = Object.keys(downloadingModels);
        if (activeDownloads.length === 0) return;

        const interval = setInterval(async () => {
            const newProgress: Record<string, number> = { ...downloadingModels };
            let changed = false;

            for (const id of activeDownloads) {
                try {
                    const p = await GetDownloadProgress(id);
                    if (p > 0 && p < 1 && p !== downloadingModels[id]) {
                        newProgress[id] = p;
                        changed = true;
                    }
                    if (p >= 1) {
                        delete newProgress[id];
                        changed = true;
                        loadStatus();
                    }
                } catch (e) {
                    console.error("Failed to fetch progress", e);
                }
            }

            if (changed) setDownloadingModels(newProgress);
        }, 1000);

        return () => clearInterval(interval);
    }, [downloadingModels]);

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
                            onClick={() => OpenConfigDir()}
                            title="Open the toolchain root directory in file explorer"
                        >
                            <Icon icon="lucide:folder-open" />
                            Browse Toolchain
                        </button>
                    </div>

                    Context-Sherpa manages a dedicated toolchain and workspace data in your `~/.context-sherpa` directory.


                    {/* Structural Analysis (ast-grep) */}
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                            <div className="flex flex-col gap-1">
                                <div className="flex items-center gap-2">
                                    <Icon icon="lucide:layout-template" className="text-primary w-5 h-5" />
                                    <h3 className="font-bold text-lg">Structural Analysis (ast-grep)</h3>
                                </div>
                                <p className="text-base-content/60 text-sm">
                                    Structural code scanning to enforce correctness and security standards.
                                </p>
                            </div>
                            {astGrepInfo?.installed ? (
                                <div className="badge badge-success badge-sm gap-1 py-2">
                                    <Icon icon="lucide:check" className="w-3 h-3" />
                                    Ready
                                </div>
                            ) : (
                                <div className="badge badge-ghost badge-sm py-2 opacity-50">Not Installed</div>
                            )}
                        </div>

                        {astGrepInfo?.installed && (
                            <div className="flex flex-col gap-1 opacity-70">
                                <span className="text-[10px] font-mono truncate">{astGrepInfo.version || 'v0.0.0'}</span>
                                <span className="text-[10px] font-mono truncate opacity-60 cursor-help" title={astGrepInfo.path}>{astGrepInfo.path}</span>
                            </div>
                        )}

                        <div className="flex items-center gap-4">
                            <button
                                className={`btn btn-sm ${astGrepInfo?.installed ? 'btn-ghost' : 'btn-primary'} ${isInstalling ? 'btn-disabled' : ''}`}
                                onClick={handleInstall}
                            >
                                {isInstalling ? (
                                    <>
                                        <span className="loading loading-spinner loading-xs"></span>
                                        Installing...
                                    </>
                                ) : (
                                    <>
                                        <Icon icon="lucide:download" className="w-4 h-4" />
                                        {astGrepInfo?.installed ? 'Update' : 'Install ast-grep'}
                                    </>
                                )}
                            </button>

                            {astGrepInfo?.installed && (
                                <DeleteAction
                                    id="ast-grep"
                                    confirmDelete={confirmDelete}
                                    onDelete={handleToolDelete}
                                />
                            )}

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
                                            <span className="text-[10px] font-mono truncate opacity-60 cursor-help" title={lang.info.path}>{lang.info.path}</span>
                                        </div>
                                    )}

                                    <div className="mt-auto pt-2 border-t border-base-200/50 flex items-center gap-2">
                                        <button
                                            className={`btn btn-sm flex-1 ${lang.info?.installed ? 'btn-ghost' : 'btn-outline btn-secondary'} ${lang.isInstalling ? 'btn-disabled' : ''}`}
                                            onClick={() => handleScipInstall(lang.id as any)}
                                        >
                                            {lang.isInstalling ? (
                                                <>
                                                    <span className="loading loading-spinner loading-xs"></span>
                                                    Installing...
                                                </>
                                            ) : (
                                                <>
                                                    <Icon icon="lucide:download" className="w-4 h-4" />
                                                    {lang.info?.installed ? 'Update' : `Install`}
                                                </>
                                            )}
                                        </button>

                                        {lang.info?.installed && (
                                            <DeleteAction
                                                id={`scip-${lang.id}`}
                                                confirmDelete={confirmDelete}
                                                onDelete={handleToolDelete}
                                                label="Delete"
                                            />
                                        )}
                                    </div>
                                    {lang.status && (
                                        <div className={`text-[10px] ${lang.status.startsWith('Error') ? 'text-error' : 'text-success'} truncate mt-1`} title={lang.status}>
                                            {lang.status}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="divider my-0 opacity-20"></div>



                    {/* Semantic Reasoning (Local SLM) */}
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center gap-2">
                            <Icon icon="lucide:brain" className="text-accent w-5 h-5" />
                            <h3 className="font-bold text-lg">Semantic Reasoning (Local SLM)</h3>
                        </div>
                        <p className="text-base-content/60 text-sm">
                            Sandboxed local models for semantic tasks like summarization and intent routing.
                        </p>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                            {curatedModels.map((model) => {
                                const localInfo = localModels.find(m => m.id === model.id);
                                const isDownloaded = !!localInfo;
                                const progress = downloadingModels[model.id];

                                return (
                                    <div key={model.id} className="border border-base-200 rounded-lg p-4 bg-base-200/30 flex flex-col gap-3">
                                        <div className="flex items-center justify-between">
                                            <div className="flex flex-col gap-0.5">
                                                <span className="font-semibold text-sm">{model.name}</span>
                                                <span className="text-[10px] opacity-60">{model.type} • {model.size}</span>
                                                {localInfo?.path && (
                                                    <span className="text-[10px] font-mono truncate opacity-60 cursor-help mt-1" title={localInfo.path}>
                                                        {localInfo.path}
                                                    </span>
                                                )}
                                            </div>
                                            {isDownloaded ? (
                                                <div className="badge badge-success badge-sm gap-1 py-2">
                                                    <Icon icon="lucide:check" className="w-3 h-3" />
                                                    Downloaded
                                                </div>
                                            ) : progress !== undefined ? (
                                                <div className="badge badge-ghost badge-sm py-2 animate-pulse">Downloading...</div>
                                            ) : (
                                                <div className="badge badge-ghost badge-sm py-2 opacity-50">Not Present</div>
                                            )}
                                        </div>

                                        {progress !== undefined && (
                                            <progress className="progress progress-accent w-full" value={progress * 100} max="100"></progress>
                                        )}

                                        <div className="mt-auto pt-2 border-t border-base-200/50">
                                            {isDownloaded ? (
                                                <DeleteAction
                                                    id={model.id}
                                                    confirmDelete={confirmDelete}
                                                    onDelete={handleToolDelete}
                                                    label="Delete"
                                                    className="w-full"
                                                />
                                            ) : (
                                                <button
                                                    className={`btn btn-sm w-full ${progress !== undefined ? 'btn-disabled' : 'btn-outline btn-accent'}`}
                                                    onClick={() => handleModelDownload(model)}
                                                >
                                                    {progress !== undefined ? (
                                                        <>
                                                            <span className="loading loading-spinner loading-xs"></span>
                                                            Downloading...
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Icon icon="lucide:download" className="w-4 h-4" />
                                                            Download Model
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

