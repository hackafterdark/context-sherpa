import { useEffect, useState } from 'react';
import { InstallAstGrep, GetAstGrepStatus, InstallScipIndexer, GetScipIndexerStatus, OpenConfigDir, DeleteAstGrep, DeleteScipIndexer, GetPreferences, SavePreferences, TestInferenceConnection } from '../wailsjs/go/main/App';
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

    const [prefs, setPrefs] = useState<any>(null);
    const [isTesting, setIsTesting] = useState(false);
    const [testStatus, setTestStatus] = useState<{ success: boolean; message: string } | null>(null);

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

            const p = await GetPreferences();
            setPrefs(p);
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

    const handlePreferenceChange = async (key: string, value: any) => {
        const newPrefs = { ...prefs, [key]: value };
        setPrefs(newPrefs);
        setTestStatus(null);

        // Auto-save if disabling inference or if value is 'disabled'
        if (value === 'disabled' || (key === 'inferenceProvider' && value === 'disabled')) {
            try {
                await SavePreferences(newPrefs);
            } catch (e) {
                console.error("Failed to auto-save disabled state:", e);
            }
        }
    };

    const handleTestConnection = async () => {
        if (!prefs) return;
        setIsTesting(true);
        setTestStatus(null);

        try {
            const provider = prefs.inferenceProvider || 'ollama';
            const url = prefs.inferenceURL || (provider === 'ollama' ? 'http://localhost:11434' : 'http://localhost:1234/v1');

            const modelName = await TestInferenceConnection(provider, url);
            setTestStatus({
                success: true,
                message: `Connected: Found ${modelName}`
            });

            // Auto-save on successful test
            await SavePreferences({
                ...prefs,
                inferenceProvider: provider,
                inferenceURL: url
            });
        } catch (e: any) {
            setTestStatus({
                success: false,
                message: (e as string).toString()
            });
        } finally {
            setIsTesting(false);
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
            }
            await loadStatus();
        } catch (e) {
            console.error("Failed to delete:", e);
            alert("Deletion failed: " + e);
        }
    };


    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-300 max-w-5xl">
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
                        <div className="flex items-center justify-between gap-4">
                            <div className="flex flex-col gap-1 min-w-0">
                                <div className="flex items-center gap-2">
                                    <Icon icon="lucide:layout-template" className="text-primary w-5 h-5" />
                                    <h3 className="font-bold text-lg">Structural Analysis (ast-grep)</h3>
                                </div>
                                <p className="text-base-content/60 text-sm">
                                    Structural code scanning to enforce correctness and security standards.
                                </p>
                            </div>
                            {astGrepInfo?.installed ? (
                                <div className="badge badge-success badge-sm gap-1 py-2 flex-shrink-0">
                                    <Icon icon="lucide:check" className="w-3 h-3" />
                                    Ready
                                </div>
                            ) : (
                                <div className="badge badge-ghost badge-sm py-2 opacity-50 flex-shrink-0">Not Installed</div>
                            )}
                        </div>

                        {astGrepInfo?.installed && (
                            <div className="flex flex-col gap-1 opacity-70 min-w-0">
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
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="font-semibold text-sm capitalize">{lang.id}</span>
                                        {lang.info?.installed ? (
                                            <div className="badge badge-success badge-sm gap-1 py-2 flex-shrink-0">
                                                <Icon icon="lucide:check" className="w-3 h-3" />
                                                Ready
                                            </div>
                                        ) : (
                                            <div className="badge badge-ghost badge-sm py-2 opacity-50 flex-shrink-0">Not Installed</div>
                                        )}
                                    </div>
                                    ...
                                    {lang.info?.installed && (
                                        <div className="flex flex-col gap-1 opacity-70 min-w-0">
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



                    {/* External Inference Provider (Ollama & LM Studio) */}
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center gap-2">
                            <Icon icon="lucide:brain" className="text-accent w-5 h-5" />
                            <h3 className="font-bold text-lg">External Inference Provider</h3>
                        </div>
                        <p className="text-base-content/60 text-sm">
                            Configure an external LLM provider like Ollama or LM Studio to enable semantic reasoning.
                        </p>

                        <div className="grid grid-cols-1 gap-4 mt-2">
                            <div className="border border-base-200 rounded-lg p-6 bg-base-200/30 flex flex-col gap-6">
                                <div className="flex flex-col gap-4">
                                    <div className="form-control">
                                        <label className="label">
                                            <span className="label-text font-semibold">Inference Engine</span>
                                        </label>
                                        <div className="flex gap-4">
                                            <label className="label cursor-pointer gap-2">
                                                <input
                                                    type="radio"
                                                    name="provider"
                                                    className="radio radio-primary radio-sm"
                                                    checked={prefs?.inferenceProvider === 'ollama'}
                                                    onChange={() => handlePreferenceChange('inferenceProvider', 'ollama')}
                                                />
                                                <span className="label-text">Ollama</span>
                                            </label>
                                            <label className="label cursor-pointer gap-2">
                                                <input
                                                    type="radio"
                                                    name="provider"
                                                    className="radio radio-primary radio-sm"
                                                    checked={prefs?.inferenceProvider === 'openai'}
                                                    onChange={() => handlePreferenceChange('inferenceProvider', 'openai')}
                                                />
                                                <span className="label-text">LM Studio (OpenAI Compatible)</span>
                                            </label>
                                            <label className="label cursor-pointer gap-2">
                                                <input
                                                    type="radio"
                                                    name="provider"
                                                    className="radio radio-primary radio-sm"
                                                    checked={prefs?.inferenceProvider === 'disabled' || !prefs?.inferenceProvider}
                                                    onChange={() => handlePreferenceChange('inferenceProvider', 'disabled')}
                                                />
                                                <span className="label-text">Disabled</span>
                                            </label>
                                        </div>
                                    </div>

                                    {prefs?.inferenceProvider !== 'disabled' && prefs?.inferenceProvider && (
                                        <div className="form-control w-full animate-in slide-in-from-top-2 duration-200">
                                            <label className="label">
                                                <span className="label-text font-semibold">Endpoint Base URL</span>
                                            </label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="text"
                                                    className="input input-bordered input-sm flex-1 font-mono"
                                                    placeholder={prefs?.inferenceProvider === 'ollama' ? "http://localhost:11434" : "http://localhost:1234/v1"}
                                                    value={prefs?.inferenceURL || ''}
                                                    onChange={(e) => handlePreferenceChange('inferenceURL', e.target.value)}
                                                />
                                                <button
                                                    className={`btn btn-sm ${testStatus?.success === true ? 'btn-success' : testStatus?.success === false ? 'btn-error' : 'btn-primary'} ${isTesting ? 'btn-disabled' : ''}`}
                                                    onClick={handleTestConnection}
                                                >
                                                    {isTesting ? (
                                                        <span className="loading loading-spinner loading-xs"></span>
                                                    ) : (
                                                        <Icon icon="lucide:zap" />
                                                    )}
                                                    {testStatus?.success ? 'Connected' : 'Test & Save'}
                                                </button>
                                            </div>
                                            {testStatus?.message && (
                                                <label className="label">
                                                    <span className={`label-text-alt ${testStatus.success ? 'text-success' : 'text-error'} font-medium`}>
                                                        {testStatus.message}
                                                    </span>
                                                </label>
                                            )}
                                        </div>
                                    )}

                                    {prefs?.inferenceProvider === 'disabled' && (
                                        <div className="p-4 bg-base-200/50 rounded-lg flex items-center justify-center border border-dashed border-base-300">
                                            <p className="text-sm opacity-50 italic">Semantic reasoning tools (query_local_reasoning, etc.) will be disabled.</p>
                                        </div>
                                    )}
                                </div>

                                <div className="p-4 bg-base-300/50 rounded-lg flex items-start gap-3 border border-base-200">
                                    <Icon icon="lucide:info" className="text-info w-5 h-5 mt-0.5 flex-shrink-0" />
                                    <div className="text-sm">
                                        <p className="font-semibold mb-1">How to connect:</p>
                                        <ul className="list-disc list-inside space-y-1 opacity-70">
                                            <li><strong>Ollama</strong>: Ensure Ollama is running. Default is <code className="text-xs">http://localhost:11434</code>.</li>
                                            <li><strong>LM Studio</strong>: Load a model and enable "Local Server". Default is <code className="text-xs">http://localhost:1234/v1</code>.</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

