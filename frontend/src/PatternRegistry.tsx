import { useState, useEffect } from 'react';
import { Icon } from '@iconify/react';
import {
    GetWorkspaces,
    SearchCommunityRules,
    GetCommunityRuleDetails,
    ImportCommunityRule,
    GetWorkspaceConfigs,
    InitializeAstGrepConfig,
    RemoveLocalRule,
    GetLocalRulesInDir,
    GetLocalRuleDetails,
    PickDirectoryWithRoot
} from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';

type RulesProps = {
    workspaceRoot: string;
    onWorkspaceChange: (root: string) => void;
};

type CommunityRule = {
    id: string;
    language: string;
    author: string;
    description: string;
    tags: string[];
};

type SgConfig = {
    id: string;
    language: string;
    path: string;
    directory: string;
    ruleDirs: string[];
};

const SUPPORTED_LANGUAGES = [
    'go', 'python', 'typescript', 'javascript', 'rust', 'cpp', 'java',
    'kotlin', 'swift', 'c', 'css', 'html', 'json', 'yaml', 'toml',
    'ruby', 'php', 'scala', 'bash', 'dart', 'elixir', 'haskell',
    'lua', 'perl', 'sql', 'zig'
];

export default function PatternRegistry({ workspaceRoot, onWorkspaceChange }: RulesProps) {
    const [searchQuery, setSearchQuery] = useState('');
    const [rules, setRules] = useState<CommunityRule[]>([]);
    const [localRules, setLocalRules] = useState<{ name: string; dir: string }[]>([]);
    const [loading, setLoading] = useState(false);
    const [selectedRule, setSelectedRule] = useState<CommunityRule | null>(null);
    const [ruleDetails, setRuleDetails] = useState<any>(null);
    const [detailsLoading, setDetailsLoading] = useState(false);
    const [importing, setImporting] = useState<string | null>(null);
    const [allWorkspaces, setAllWorkspaces] = useState<any[]>([]);

    // New phase 5 state
    const [configs, setConfigs] = useState<SgConfig[]>([]);
    const [selectedConfig, setSelectedConfig] = useState<SgConfig | null>(null);
    const [isInitModalOpen, setIsInitModalOpen] = useState(false);
    const [initPath, setInitPath] = useState('');
    const [initLang, setInitLang] = useState('go');
    const [isInitializing, setIsInitializing] = useState(false);

    // Phase 5 additions
    const [localRuleDetails, setLocalRuleDetails] = useState<any>(null);
    const [isLocalDetailsLoading, setIsLocalDetailsLoading] = useState(false);
    const [ruleToDelete, setRuleToDelete] = useState<{ name: string; dir: string } | null>(null);
    const [selectedTag, setSelectedTag] = useState('');

    useEffect(() => {
        const fetchWorkspaces = async () => {
            try {
                const ws = await GetWorkspaces();
                const unique = Array.from(new Map((ws as any[]).map(item => [item.root, item])).values());
                setAllWorkspaces(unique);
            } catch (e) {
                console.error("Error fetching workspaces:", e);
            }
        };

        fetchWorkspaces();
        EventsOn('workspace-updated', (ws: any[]) => {
            const unique = Array.from(new Map(ws.map(item => [item.root, item])).values());
            setAllWorkspaces(unique);
        });

        return () => EventsOff('workspace-updated');
    }, []);

    useEffect(() => {
        if (workspaceRoot) {
            discoverConfigs();
            setInitPath(workspaceRoot);
        }
    }, [workspaceRoot]);

    useEffect(() => {
        if (selectedConfig) {
            fetchLocalRules();
            handleSearch();
        } else {
            setLocalRules([]);
            setRules([]);
        }
    }, [selectedConfig]);

    const discoverConfigs = async () => {
        try {
            const discovered = await GetWorkspaceConfigs(workspaceRoot) as SgConfig[];
            setConfigs(discovered || []);
            if (discovered && discovered.length > 0) {
                // Keep selection if it still exists, otherwise pick first
                const stillExists = discovered.find(c => c.path === selectedConfig?.path);
                if (!stillExists) {
                    setSelectedConfig(discovered[0]);
                } else {
                    setSelectedConfig(stillExists);
                }
            } else {
                setSelectedConfig(null);
            }
        } catch (e) {
            console.error("Error discovering configs:", e);
        }
    };

    const handleSearch = async () => {
        if (!selectedConfig) return;
        setLoading(true);
        try {
            // Auto-filter by language based on selected config, and include selectedTag
            const results = await SearchCommunityRules(searchQuery, selectedConfig.language, selectedTag);
            setRules(results || []);
        } catch (e) {
            console.error("Error searching rules:", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        handleSearch();
    }, [selectedTag]);

    const fetchLocalRules = async () => {
        if (!selectedConfig) return;
        try {
            let allRules: { name: string; dir: string }[] = [];
            for (const rd of selectedConfig.ruleDirs) {
                const fullRuleDir = `${selectedConfig.directory}/${rd}`;
                const dirRules = await GetLocalRulesInDir(fullRuleDir);
                if (dirRules) {
                    allRules = [...allRules, ...dirRules.map(name => ({ name, dir: fullRuleDir }))];
                }
            }
            setLocalRules(allRules);
        } catch (e) {
            console.error("Error fetching local rules:", e);
        }
    };

    const showRuleDetails = async (rule: CommunityRule) => {
        setSelectedRule(rule);
        setDetailsLoading(true);
        try {
            const details = await GetCommunityRuleDetails(rule.id);
            setRuleDetails(details);
        } catch (e) {
            console.error("Error fetching rule details:", e);
        } finally {
            setDetailsLoading(false);
        }
    };

    const handleImport = async (ruleID: string) => {
        if (!selectedConfig) return;
        setImporting(ruleID);
        try {
            await ImportCommunityRule(selectedConfig.path, ruleID);
            await fetchLocalRules();
        } catch (e) {
            console.error("Error importing rule:", e);
        } finally {
            setImporting(null);
        }
    };

    const handleRemove = async (ruleName: string) => {
        if (!selectedConfig) return;
        try {
            await RemoveLocalRule(selectedConfig.path, ruleName.replace(/\.ya?ml$/, ''));
            setRuleToDelete(null);
            fetchLocalRules();
        } catch (e) {
            console.error("Error removing rule:", e);
        }
    };

    const handleInspectLocalRule = async (rule: { name: string; dir: string }) => {
        setIsLocalDetailsLoading(true);
        setLocalRuleDetails(null);
        try {
            const path = rule.dir + '/' + rule.name;
            const details = await GetLocalRuleDetails(path);
            setLocalRuleDetails(details);
        } catch (e) {
            console.error("Error fetching local rule details:", e);
        } finally {
            setIsLocalDetailsLoading(false);
        }
    };

    const handleBrowse = async () => {
        try {
            const selected = await PickDirectoryWithRoot(workspaceRoot);
            if (selected) {
                setInitPath(selected);
            }
        } catch (e) {
            console.error("Error picking directory:", e);
        }
    };

    const handleInitialize = async () => {
        setIsInitializing(true);
        try {
            await InitializeAstGrepConfig(initPath, initLang);
            await discoverConfigs();
            setIsInitModalOpen(false);
        } catch (e) {
            console.error("Error initializing config:", e);
            alert("Error: " + e);
        } finally {
            setIsInitializing(false);
        }
    };

    const isRuleInstalled = (ruleID: string) => {
        return localRules.some(lr => lr.name.startsWith(ruleID));
    };
    useEffect(() => {
        const timer = setTimeout(() => {
            if (selectedConfig) {
                handleSearch();
            }
        }, 400);
        return () => clearTimeout(timer);
    }, [searchQuery, selectedConfig]);

    return (
        <div className="flex flex-col flex-1 h-full w-full animate-in fade-in duration-300">
            {/* Top Header - Unified with other sections */}
            <div className="flex justify-between items-center shrink-0">
                <div className="flex items-center gap-4">
                    <h1 className="text-3xl font-bold font-sans tracking-tight text-base-content">Pattern Registry</h1>
                    <div className="dropdown dropdown-bottom">
                        <div tabIndex={0} role="button" className="badge badge-outline badge-md opacity-60 font-mono border-base-content/10 cursor-pointer hover:bg-base-300 transition-all flex items-center gap-2 pr-1.5 h-7 rounded-lg group">
                            <span className="truncate max-w-[200px] text-[10px] font-bold">{workspaceRoot}</span>
                            <Icon icon="lucide:chevron-down" className="w-3 h-3 opacity-30 group-hover:opacity-100 transition-opacity" />
                        </div>
                        <ul tabIndex={0} className="dropdown-content z-[100] menu p-2 shadow-2xl bg-base-100 rounded-2xl border border-base-300 w-full min-w-[320px] max-w-[480px] mt-3 animate-in fade-in slide-in-from-top-2 duration-200 overflow-hidden">
                            <div className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 px-4 py-3 border-b border-base-300/50 mb-1">Loaded Workspaces</div>
                            <div className="max-h-80 overflow-y-auto scrollbar-hide py-1">
                                {allWorkspaces.map(ws => (
                                    <li key={ws.root}>
                                        <button
                                            onClick={() => {
                                                onWorkspaceChange(ws.root);
                                                (document.activeElement as HTMLElement)?.blur();
                                            }}
                                            className={`flex flex-col items-start gap-1 py-3 px-4 rounded-xl mx-1 my-0.5 ${ws.root === workspaceRoot ? 'bg-primary/10 text-primary border border-primary/20 hover:bg-primary/20' : 'hover:bg-base-200 border border-transparent'}`}
                                        >
                                            <div className="flex items-center gap-2.5 w-full">
                                                <Icon icon="lucide:folder-tree" className={`w-3.5 h-3.5 ${ws.root === workspaceRoot ? 'text-primary' : 'opacity-40'}`} />
                                                <span className="font-bold text-xs truncate flex-1 leading-none">{ws.root.split(/[\\/]/).pop() || ws.root}</span>
                                            </div>
                                            <span className="text-[10px] opacity-30 font-mono truncate w-full pl-6 leading-none">
                                                {ws.root}
                                            </span>
                                        </button>
                                    </li>
                                ))}
                            </div>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="flex flex-1 overflow-hidden min-h-0">
                {/* Sidebar for Config Selection */}
                <div className="w-64 border-r border-base-200 flex flex-col shrink-0 bg-base-200/20 pr-4 mt-4 overflow-y-auto scrollbar-hide">
                    <div className="flex items-center justify-between px-1 mb-4">
                        <h2 className="text-[10px] font-black uppercase tracking-widest opacity-40">Configurations</h2>
                        <button
                            className="btn btn-ghost btn-xs btn-circle opacity-40 hover:opacity-100"
                            onClick={() => setIsInitModalOpen(true)}
                        >
                            <Icon icon="lucide:plus" className="w-4 h-4" />
                        </button>
                    </div>
                    <div className="flex flex-col gap-1">
                        {configs.map(config => (
                            <button
                                key={config.path}
                                onClick={() => setSelectedConfig(config)}
                                className={`flex flex-col items-start p-3 rounded-xl transition-all border ${selectedConfig?.path === config.path
                                    ? 'bg-primary/10 border-primary/20 text-primary shadow-sm'
                                    : 'hover:bg-base-200 border-transparent text-base-content/70'
                                    }`}
                            >
                                <div className="flex items-center gap-2 w-full">
                                    <Icon icon="lucide:file-cog" className="shrink-0 w-3.5 h-3.5" />
                                    <span className="text-xs font-bold truncate flex-1">{config.id || 'Unnamed Config'}</span>
                                    <div className="badge badge-xs badge-neutral opacity-50 uppercase text-[8px] font-bold">{config.language}</div>
                                </div>
                                <span className="text-[10px] opacity-40 truncate w-full mt-1.5 font-mono leading-none">
                                    {config.path.replace(workspaceRoot, '').replace(/^[\\/]/, '') || './'}
                                </span>
                            </button>
                        ))}
                        {configs.length === 0 && (
                            <div className="py-20 text-center opacity-20">
                                <Icon icon="lucide:search-slash" className="mx-auto w-8 h-8 mb-2" />
                                <p className="text-[10px] font-bold uppercase tracking-widest">No configs found</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col min-w-0 overflow-hidden bg-base-200/10 rounded-2xl border border-base-200 shadow-sm mt-4">
                    {!selectedConfig ? (
                        <div className="flex-1 flex flex-col items-center justify-center p-20 text-center opacity-30">
                            <Icon icon="lucide:scroll-text" className="w-24 h-24 mb-6" />
                            <h2 className="text-2xl font-black mb-2 tracking-tight">Pattern Registry</h2>
                            <p className="max-w-md text-sm">Select or create an ast-grep configuration in the sidebar to start managing patterns.</p>
                        </div>
                    ) : (
                        <>
                            {/* Active Config Header */}
                            <div className="px-8 py-6 border-b border-base-300 shrink-0 flex justify-between items-center bg-base-100/50">
                                <div className="flex items-center gap-4">
                                    <div>
                                        <h2 className="text-xl font-black tracking-tight flex items-center gap-3">
                                            {selectedConfig.id}
                                            <div className="badge badge-sm badge-primary uppercase font-black text-[10px] tracking-widest">{selectedConfig.language}</div>
                                        </h2>
                                        <p className="text-[11px] opacity-60 font-mono mt-1 flex items-center gap-1.5">
                                            <Icon icon="lucide:map-pin" className="w-3 h-3 opacity-40" />
                                            {selectedConfig.path}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Content Split */}
                            <div className="flex-1 flex overflow-hidden">
                                {/* Community Rules (Main) */}
                                <div className="flex-1 overflow-y-auto px-4 py-4 border-r border-base-300 scrollbar-hide">
                                    <div className="flex items-center justify-between mb-8">
                                        <div className="flex items-center gap-3">
                                            <Icon icon="lucide:globe" className="opacity-20 w-5 h-5" />
                                            <h3 className="text-xs font-black uppercase tracking-[0.2em] opacity-40">Community Patterns</h3>
                                        </div>

                                        {/* Live Search Bar */}
                                        <div className="flex items-center gap-3 min-w-[400px]">
                                            {selectedTag && (
                                                <button
                                                    className="badge badge-primary badge-outline badge-lg gap-2 font-black uppercase text-[10px] tracking-widest shrink-0 h-10 px-4 group hover:bg-primary hover:text-white transition-all border-dashed"
                                                    onClick={() => setSelectedTag('')}
                                                >
                                                    <Icon icon="lucide:tag" className="w-3.5 h-3.5" />
                                                    {selectedTag}
                                                    <Icon icon="lucide:x" className="w-3 h-3 ml-1 group-hover:scale-125 transition-transform" />
                                                </button>
                                            )}
                                            <div className="relative group flex-1">
                                                <input
                                                    type="text"
                                                    placeholder="Search ast-grep rules..."
                                                    className="input input-bordered w-full bg-base-100 h-10 text-xs border-base-300 rounded-xl pl-10 focus:border-primary/50 transition-all shadow-sm"
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                />
                                                <Icon icon="lucide:search" className="absolute left-3.5 top-1/2 -translate-y-1/2 opacity-20 group-focus-within:opacity-100 transition-opacity w-4 h-4" />
                                                {loading && (
                                                    <div className="absolute right-3.5 top-1/2 -translate-y-1/2">
                                                        <span className="loading loading-spinner loading-xs text-primary/50"></span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                                        {rules.map(rule => (
                                            <div key={rule.id} className="card bg-base-100 shadow-md border border-base-200 hover:border-primary/30 hover:shadow-xl transition-all duration-300 group overflow-hidden min-w-[340px]">
                                                <div className="card-body p-6 gap-4">
                                                    <div className="flex justify-between items-start gap-4">
                                                        <h2 className="text-sm font-black truncate leading-tight group-hover:text-primary transition-colors tracking-tight">{rule.id}</h2>
                                                        {isRuleInstalled(rule.id) && (
                                                            <div className="badge badge-success badge-xs gap-1 text-[8px] font-black uppercase tracking-tighter shrink-0 opacity-80 h-5 px-2">
                                                                <Icon icon="lucide:check" className="w-2 h-2" />
                                                                Installed
                                                            </div>
                                                        )}
                                                    </div>
                                                    <p className="text-[13px] opacity-70 leading-relaxed line-clamp-2 h-10">{rule.description}</p>
                                                    <div className="flex gap-1.5 flex-wrap max-h-5 overflow-hidden mt-1">
                                                        {rule.tags?.map(tag => (
                                                            <button
                                                                key={tag}
                                                                className={`px-2 py-0.5 rounded-md text-[10px] font-black uppercase tracking-widest leading-none transition-all hover:scale-105 active:scale-95 ${selectedTag === tag ? 'bg-primary text-white opacity-100' : 'bg-base-200 opacity-40 hover:opacity-100 animate-pulse'}`}
                                                                style={{ animationDuration: '3s' }}
                                                                onClick={() => setSelectedTag(tag === selectedTag ? '' : tag)}
                                                            >
                                                                #{tag}
                                                            </button>
                                                        ))}
                                                    </div>
                                                    <div className="flex items-center justify-end mt-2 pt-4 border-t border-base-200/50">
                                                        <div className="card-actions justify-end gap-2">
                                                            <button
                                                                className="btn btn-ghost btn-xs font-black uppercase tracking-widest text-[10px] h-8 px-3 rounded-lg"
                                                                onClick={() => showRuleDetails(rule)}
                                                            >
                                                                Details
                                                            </button>
                                                            {!isRuleInstalled(rule.id) && (
                                                                <button
                                                                    className="btn btn-primary btn-xs font-black uppercase tracking-widest text-[10px] h-8 px-4 rounded-lg bg-primary hover:bg-primary-focus border-none"
                                                                    onClick={() => handleImport(rule.id)}
                                                                    disabled={importing === rule.id}
                                                                >
                                                                    {importing === rule.id ? <span className="loading loading-spinner loading-xs"></span> : "Import"}
                                                                </button>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                        {rules.length === 0 && !loading && (
                                            <div className="col-span-full py-32 text-center opacity-20 border border-dashed border-base-300 rounded-[2rem]">
                                                <Icon icon="lucide:search-x" className="text-7xl mx-auto mb-6" />
                                                <p className="text-xl font-black uppercase tracking-[0.2em]">No patterns matched</p>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Local Rules (Side) */}
                                <div className="w-80 overflow-y-auto px-4 py-4 bg-black/5 flex flex-col scrollbar-hide">
                                    <div className="flex items-center gap-3 mb-4 sticky top-0 bg-transparent backdrop-blur-md z-20 py-1">
                                        <div className="p-1.5 bg-success/10 text-success rounded-lg">
                                            <Icon icon="lucide:package-check" className="w-4 h-4" />
                                        </div>
                                        <h3 className="text-xs font-black uppercase tracking-[0.2em] opacity-40">Local Patterns</h3>
                                        <div className="badge badge-ghost opacity-40 ml-auto font-mono text-[11px] font-bold border-none">{localRules.length}</div>
                                    </div>
                                    <div className="flex flex-col gap-3">
                                        {localRules.map(rule => (
                                            <div key={rule.name + rule.dir} className="flex items-center justify-between p-4 bg-base-100 rounded-2xl border border-base-200 shadow-sm group hover:border-primary/20 transition-all duration-300">
                                                <div className="flex flex-col min-w-0 pr-3">
                                                    <span className="text-xs font-black truncate leading-tight tracking-tight">{rule.name.replace(/\.ya?ml$/, '')}</span>
                                                    <span className="text-[10px] opacity-30 font-mono truncate mt-1">{rule.dir.replace(workspaceRoot, '').replace(/^[\\/]/, '') || './'}</span>
                                                </div>
                                                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button
                                                        className="btn btn-ghost btn-xs btn-square text-primary/60 hover:text-primary"
                                                        onClick={() => handleInspectLocalRule(rule)}
                                                        title="Inspect Rule"
                                                    >
                                                        <Icon icon="lucide:eye" className="w-4 h-4" />
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs btn-square text-error/60 hover:text-error"
                                                        onClick={() => setRuleToDelete(rule)}
                                                        title="Delete Rule"
                                                    >
                                                        <Icon icon="lucide:trash-2" className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                        {localRules.length === 0 && (
                                            <div className="py-16 text-center opacity-30 bg-base-100/50 rounded-3xl border border-dashed border-base-300">
                                                <Icon icon="lucide:archive-x" className="mx-auto w-12 h-12 mb-4 opacity-50 text-base-content/20" />
                                                <p className="text-[10px] font-black uppercase tracking-widest text-base-content/50">Workspace Empty</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Rule Details Modal */}
            {selectedRule && (
                <div className="modal modal-open">
                    <div className="modal-box w-11/12 max-w-5xl h-[85vh] flex flex-col p-0 overflow-hidden shadow-2xl border border-base-300 rounded-3xl">
                        <div className="flex justify-between items-center p-5 border-b border-base-300 bg-base-200">
                            <div className="flex items-center gap-4">
                                <div className="badge badge-lg badge-outline opacity-40 uppercase font-black text-[10px] tracking-widest">{selectedRule.language}</div>
                                <div>
                                    <h3 className="font-black text-xl leading-none">{selectedRule.id}</h3>
                                    <p className="text-[10px] opacity-50 font-bold uppercase tracking-widest mt-1.5 flex items-center gap-1">
                                        <Icon icon="lucide:user" className="w-2.5 h-2.5" />
                                        {selectedRule.author}
                                    </p>
                                </div>
                            </div>
                            <button className="btn btn-sm btn-circle btn-ghost" onClick={() => setSelectedRule(null)}>
                                <Icon icon="lucide:x" className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-8 bg-base-100">
                            <div>
                                <h4 className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Overview</h4>
                                <div className="p-5 bg-base-200/50 rounded-2xl border border-base-300 text-sm leading-relaxed font-medium">
                                    {selectedRule.description}
                                </div>
                            </div>
                            <div>
                                <h4 className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Tags</h4>
                                <div className="flex gap-2 flex-wrap pb-2">
                                    {selectedRule.tags?.map(tag => (
                                        <button
                                            key={tag}
                                            className={`px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-wider transition-all hover:scale-105 active:scale-95 ${selectedTag === tag ? 'bg-primary text-white shadow-lg shadow-primary/30' : 'bg-base-200 opacity-60 hover:opacity-100 border border-base-300'}`}
                                            onClick={() => {
                                                setSelectedTag(tag);
                                                setSelectedRule(null);
                                            }}
                                        >
                                            #{tag}
                                        </button>
                                    ))}
                                    {(!selectedRule.tags || selectedRule.tags.length === 0) && (
                                        <span className="text-[10px] opacity-20 italic font-medium ml-1">No tags available</span>
                                    )}
                                </div>
                            </div>
                            <div className="flex-1 flex flex-col min-h-0">
                                <div className="flex items-center justify-between mb-3 px-1">
                                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30">Pattern Source</h4>
                                    {ruleDetails?.content && (
                                        <button
                                            className="btn btn-ghost btn-xs gap-1.5 opacity-40 hover:opacity-100"
                                            onClick={() => {
                                                navigator.clipboard.writeText(ruleDetails.content);
                                            }}
                                        >
                                            <Icon icon="lucide:copy" className="w-3 h-3" />
                                            Copy
                                        </button>
                                    )}
                                </div>
                                {detailsLoading ? (
                                    <div className="flex-1 min-h-[400px] flex items-center justify-center bg-base-200 rounded-2xl border border-base-300">
                                        <span className="loading loading-spinner loading-md opacity-20"></span>
                                    </div>
                                ) : (
                                    <div className="flex-1 min-h-[400px] bg-base-300 rounded-2xl overflow-hidden border border-base-content/10 shadow-inner group relative">
                                        <pre className="p-6 text-xs font-mono h-full overflow-auto whitespace-pre scrollbar-hide">
                                            <code className="text-base-content/90">{ruleDetails?.content}</code>
                                        </pre>
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="p-5 border-t border-base-300 flex justify-end gap-3 bg-base-200">
                            <button className="btn btn-ghost btn-sm font-bold px-6" onClick={() => setSelectedRule(null)}>Close</button>
                            {!isRuleInstalled(selectedRule.id) && (
                                <button
                                    className="btn btn-primary btn-sm px-8 font-black uppercase tracking-widest text-[10px]"
                                    onClick={() => handleImport(selectedRule.id)}
                                    disabled={importing === selectedRule.id}
                                >
                                    {importing === selectedRule.id ? <span className="loading loading-spinner loading-xs"></span> : "Import Pattern"}
                                </button>
                            )}
                        </div>
                    </div>
                    <div className="modal-backdrop bg-base-300/60 backdrop-blur-sm" onClick={() => setSelectedRule(null)}></div>
                </div>
            )}

            {/* Initialization Modal */}
            {isInitModalOpen && (
                <div className="modal modal-open">
                    <div className="modal-box max-w-md bg-base-100 border border-base-300 rounded-3xl p-8 shadow-2xl">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="w-12 h-12 rounded-2xl bg-primary/20 flex items-center justify-center text-primary">
                                <Icon icon="lucide:sparkles" className="w-6 h-6" />
                            </div>
                            <div>
                                <h3 className="text-xl font-black leading-tight">Initialize ast-grep</h3>
                                <p className="text-xs opacity-40 font-bold uppercase tracking-widest mt-0.5">Setup pattern management</p>
                            </div>
                        </div>

                        <div className="space-y-6">
                            <div className="form-control w-full">
                                <label className="label py-1">
                                    <span className="label-text text-[11px] font-black uppercase tracking-widest opacity-40">Target Directory</span>
                                </label>
                                <div className="relative group flex gap-2">
                                    <div className="relative flex-1 group">
                                        <Icon icon="lucide:folder-open" className="absolute left-3.5 top-1/2 -translate-y-1/2 opacity-30 group-focus-within:opacity-100 text-primary transition-opacity" />
                                        <input
                                            type="text"
                                            className="input input-bordered w-full pl-11 h-12 text-sm font-mono border-base-300 focus:border-primary/50 transition-all rounded-xl"
                                            value={initPath}
                                            onChange={(e) => setInitPath(e.target.value)}
                                            placeholder="Full path to directory..."
                                        />
                                    </div>
                                    <button
                                        className="btn btn-square btn-ghost h-12 w-12 border border-base-300 rounded-xl hover:bg-base-200"
                                        onClick={handleBrowse}
                                        title="Browse directory"
                                    >
                                        <Icon icon="lucide:search" className="w-5 h-5 opacity-40" />
                                    </button>
                                </div>
                                <label className="label py-0.5 px-1 mt-1">
                                    <span className="label-text-alt opacity-30 text-[12px]">Will create <b>sgconfig.yml</b> and <b>rules/</b></span>
                                </label>
                            </div>

                            <div className="form-control w-full">
                                <label className="label py-1">
                                    <span className="label-text text-[11px] font-black uppercase tracking-widest opacity-40">Primary Language</span>
                                </label>
                                <div className="relative group">
                                    <select
                                        className="select select-bordered w-full h-12 text-sm border-base-300 focus:border-primary/50 transition-all rounded-xl pl-4"
                                        value={initLang}
                                        onChange={(e) => setInitLang(e.target.value)}
                                    >
                                        {SUPPORTED_LANGUAGES.map(lang => (
                                            <option key={lang} value={lang}>{lang.toUpperCase()}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="modal-action mt-10 gap-2">
                            <button
                                className="btn btn-ghost font-bold flex-1 h-12 rounded-xl"
                                onClick={() => setIsInitModalOpen(false)}
                                disabled={isInitializing}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary font-black uppercase tracking-widest text-[11px] flex-1 h-12 rounded-xl"
                                onClick={handleInitialize}
                                disabled={isInitializing || !initPath}
                            >
                                {isInitializing ? <span className="loading loading-spinner loading-sm"></span> : "Initialize"}
                            </button>
                        </div>
                    </div>
                    <div className="modal-backdrop bg-base-300/40 backdrop-blur-sm" onClick={() => !isInitializing && setIsInitModalOpen(false)}></div>
                </div>
            )}
            {/* Local Rule Details Modal */}
            {localRuleDetails && (
                <div className="modal modal-open">
                    <div className="modal-box w-11/12 max-w-5xl h-[85vh] flex flex-col p-0 overflow-hidden shadow-2xl border border-base-300 rounded-3xl">
                        <div className="flex justify-between items-center p-5 border-b border-base-300 bg-base-200">
                            <div className="flex items-center gap-4">
                                <div className="badge badge-lg badge-outline opacity-40 uppercase font-black text-[10px] tracking-widest">{localRuleDetails.language || 'generic'}</div>
                                <div>
                                    <h3 className="font-black text-xl leading-none">{localRuleDetails.id || 'Unnamed Rule'}</h3>
                                    <p className="text-[10px] opacity-50 font-bold uppercase tracking-widest mt-1.5 flex items-center gap-1">
                                        <Icon icon="lucide:alert-circle" className="w-2.5 h-2.5" />
                                        Severity: {localRuleDetails.severity || 'n/a'}
                                    </p>
                                </div>
                            </div>
                            <button className="btn btn-sm btn-circle btn-ghost" onClick={() => setLocalRuleDetails(null)}>
                                <Icon icon="lucide:x" className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-8 bg-base-100 font-sans">
                            {localRuleDetails.message && (
                                <div>
                                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Message</h4>
                                    <div className="p-5 bg-base-200/50 rounded-2xl border border-base-300 text-sm leading-relaxed font-medium">
                                        {localRuleDetails.message}
                                    </div>
                                </div>
                            )}
                            <div className="flex-1 flex flex-col min-h-0">
                                <div className="flex items-center justify-between mb-3 px-1">
                                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30">Full YAML</h4>
                                    <button
                                        className="btn btn-ghost btn-xs gap-1.5 opacity-40 hover:opacity-100"
                                        onClick={() => {
                                            navigator.clipboard.writeText(localRuleDetails.content);
                                        }}
                                    >
                                        <Icon icon="lucide:copy" className="w-3 h-3" />
                                        Copy
                                    </button>
                                </div>
                                <div className="flex-1 min-h-[400px] bg-base-300 rounded-2xl overflow-hidden border border-base-content/10 shadow-inner relative">
                                    <pre className="p-6 text-xs font-mono h-full overflow-auto whitespace-pre scrollbar-hide">
                                        <code className="text-base-content/90">{localRuleDetails.content}</code>
                                    </pre>
                                </div>
                            </div>
                        </div>
                        <div className="p-5 border-t border-base-300 flex justify-end gap-3 bg-base-200">
                            <button className="btn btn-ghost btn-sm font-bold px-6" onClick={() => setLocalRuleDetails(null)}>Close</button>
                        </div>
                    </div>
                    <div className="modal-backdrop bg-base-300/60 backdrop-blur-sm" onClick={() => setLocalRuleDetails(null)}></div>
                </div>
            )}

            {/* Inspect Loading State */}
            {isLocalDetailsLoading && (
                <div className="modal modal-open">
                    <div className="modal-box max-w-xs bg-base-100 py-10 text-center rounded-3xl">
                        <span className="loading loading-spinner loading-lg text-primary opacity-20"></span>
                        <p className="text-[10px] font-black uppercase tracking-widest opacity-40 mt-4">Loading rule details...</p>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {ruleToDelete && (
                <div className="modal modal-open">
                    <div className="modal-box max-w-sm bg-base-100 border border-base-300 rounded-3xl p-8 shadow-2xl">
                        <div className="flex flex-col items-center text-center">
                            <div className="w-16 h-16 rounded-full bg-error/10 flex items-center justify-center text-error mb-4">
                                <Icon icon="lucide:alert-triangle" className="w-8 h-8" />
                            </div>
                            <h3 className="text-xl font-black">Delete Rule?</h3>
                            <p className="text-xs opacity-60 font-medium mt-2">
                                Are you sure you want to remove <b className="text-base-content">{ruleToDelete.name.replace(/\.ya?ml$/, '')}</b>?
                                This action cannot be undone.
                            </p>
                        </div>
                        <div className="flex gap-3 mt-8">
                            <button
                                className="btn btn-ghost flex-1 font-bold h-12 rounded-xl"
                                onClick={() => setRuleToDelete(null)}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-error flex-1 font-black uppercase tracking-widest text-[11px] h-12 rounded-xl"
                                onClick={() => handleRemove(ruleToDelete.name)}
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                    <div className="modal-backdrop bg-base-300/40 backdrop-blur-sm" onClick={() => setRuleToDelete(null)}></div>
                </div>
            )}
        </div>
    );
}
