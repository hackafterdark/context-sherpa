import { useState, useEffect, useRef } from 'react';
import { Icon } from '@iconify/react';
import { ReadMarkdown, WriteMarkdown, GetWorkspaces } from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';
import MarkdownEditor from './components/MarkdownEditor';
import SkillCatalog from './components/SkillCatalog';

type SkillsProps = {
    workspaceRoot: string;
    onWorkspaceChange: (root: string) => void;
    theme: string;
};

const DRAFT_PREFIX = 'hub-skill-draft:';

const normalizeText = (text: string) => {
    if (!text) return '';
    // Strip BOM, normalize ALL carriage returns (\r\n or just \r) to \n
    return text
        .replace(/^\uFEFF/, '')                 // Strip BOM
        .replace(/\r\n|\r/g, '\n')              // Normalize line endings
        .trim();                                // Trim flanking whitespace
};

export default function Skills({ workspaceRoot, onWorkspaceChange, theme }: SkillsProps) {
    const [currentPath, setCurrentPath] = useState('');
    const [content, setContent] = useState('');
    const [originalContent, setOriginalContent] = useState('');
    const [isDirty, setIsDirty] = useState(false);
    const [saving, setSaving] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [pendingPath, setPendingPath] = useState<string | null>(null);
    const [showConfirmModal, setShowConfirmModal] = useState(false);
    const [allWorkspaces, setAllWorkspaces] = useState<any[]>([]);
    const lastLoadTimeRef = useRef(0);

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

    const loadFile = async (path: string, force = false) => {
        // Return early if already loading or if clicking the same file (unless forced)
        if (isLoading) return;
        if (!force && path.toLowerCase() === currentPath.toLowerCase()) return;

        if (isDirty) {
            setPendingPath(path);
            setShowConfirmModal(true);
            return;
        }

        setIsLoading(true);
        console.log("Hub: Loading file:", path);
        try {
            const rawData = await ReadMarkdown(path);
            console.log(`Hub: ReadMarkdown success. Path: ${path}, Bytes: ${rawData?.length || 0}`);
            const data = normalizeText(rawData);
            console.log(`Hub: Normalized content string length: ${data.length}`);

            if (data.length === 0 && rawData.length > 0) {
                console.warn("Hub: Normalization resulted in EMPTY string but raw data existed!", { raw: rawData.substring(0, 50) });
            }

            const normPath = path.toLowerCase();

            // Check for persistent draft
            const rawDraft = localStorage.getItem(DRAFT_PREFIX + normPath);
            const draft = rawDraft ? normalizeText(rawDraft) : null;

            // Set fundamental states FIRST
            console.log("Hub: Setting originalContent and currentPath...");
            console.log('data', data);
            console.log('draft', draft);
            setOriginalContent(data);
            setCurrentPath(path);

            if (draft && draft !== data) {
                console.log("Hub: Restoring draft for", path);
                setContent(rawDraft!);
                setIsDirty(true);
            } else {
                console.log("Hub: Setting clean content for", path);
                setContent(data);
                setIsDirty(false);
                if (rawDraft) localStorage.removeItem(DRAFT_PREFIX + normPath);
            }

            // Set stabilization window for normalization
            lastLoadTimeRef.current = Date.now();
        } catch (err) {
            console.error("Hub: Load failed for path:", path, err);
            setCurrentPath('');
            setContent('');
            setIsDirty(false);
        } finally {
            console.log("Hub: Finalizing load for", path);
            setTimeout(() => setIsLoading(false), 150);
        }
    };

    const handleContentChange = (newContent: string) => {
        const normalizedInput = normalizeText(newContent);

        // STABILIZATION WINDOW: MDXEditor often normalizes content on load.
        // We allow a 3s window to sync originalContent with the editor's internal format.
        const timeSinceLoad = Date.now() - lastLoadTimeRef.current;
        if (timeSinceLoad < 3000 && !isDirty) {
            if (normalizedInput !== originalContent) {
                // Safeguard: If editor fires an empty string on mount but we have real data, disregard.
                if (normalizedInput === "" && originalContent !== "") {
                    console.warn(`Hub: Stabilization ignored empty input for non-empty file (T+${timeSinceLoad}ms)`);
                    return;
                }

                console.log(`Hub: Stabilization sync (T+${timeSinceLoad}ms). Syncing originalContent.`);
                setOriginalContent(normalizedInput);
                setContent(newContent);
                return;
            }
        }

        // Normal path
        setContent(newContent);

        const dirty = normalizedInput !== originalContent;
        if (dirty !== isDirty) {
            console.log("Hub: Dirty state:", dirty, {
                currentLen: normalizedInput.length,
                originalLen: originalContent.length
            });
        }
        setIsDirty(dirty);

        if (currentPath) {
            const normPath = currentPath.toLowerCase();
            if (dirty) {
                localStorage.setItem(DRAFT_PREFIX + normPath, newContent);
            } else {
                localStorage.removeItem(DRAFT_PREFIX + normPath);
            }
        }
    };

    const handleSave = async () => {
        if (!currentPath || !isDirty) return;

        setSaving(true);
        try {
            await WriteMarkdown(currentPath, content);
            const normalizedSaved = normalizeText(content);
            setOriginalContent(normalizedSaved);
            setIsDirty(false);
            localStorage.removeItem(DRAFT_PREFIX + currentPath.toLowerCase());
        } catch (err) {
            console.error("Failed to save markdown:", err);
        } finally {
            setSaving(false);
        }
    };

    const handleDiscard = () => {
        setContent(originalContent);
        setIsDirty(false);
        if (currentPath) {
            localStorage.removeItem(DRAFT_PREFIX + currentPath.toLowerCase());
        }
    };

    const confirmDiscardModal = async () => {
        if (pendingPath) {
            const nextPath = pendingPath;
            localStorage.removeItem(DRAFT_PREFIX + currentPath.toLowerCase());
            setIsDirty(false);
            setPendingPath(null);
            setShowConfirmModal(false);

            // Re-load the next file using the clean loadFile logic
            loadFile(nextPath);
        }
    };

    return (
        <div className="flex flex-col flex-1 h-full w-full gap-4 animate-in fade-in duration-300 bg-base-100">
            {/* Main Header matching Atlas.tsx */}
            <div className="flex justify-between items-center px-6 pt-5 shrink-0">
                <div className="flex items-center gap-4">
                    <h1 className="text-3xl font-bold font-sans tracking-tight">Behavior Adjustment</h1>
                    <div className="dropdown dropdown-bottom">
                        <div tabIndex={0} role="button" className="badge badge-outline badge-md opacity-60 font-mono border-base-content/10 cursor-pointer hover:bg-base-300 transition-all flex items-center gap-2 pr-1.5 h-7 rounded-lg group">
                            <span className="truncate max-w-[200px] text-[10px] font-bold">{workspaceRoot}</span>
                            <Icon icon="lucide:chevron-down" className="w-3 h-3 opacity-30 group-hover:opacity-100 transition-opacity" />
                        </div>
                        <ul tabIndex={0} className="dropdown-content z-[100] menu p-2 shadow-2xl bg-base-100 rounded-2xl border border-base-300 w-96 mt-3 animate-in fade-in slide-in-from-top-2 duration-200 overflow-hidden">
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

            {/* Viewport Content */}
            <div className="flex flex-1 gap-0 pb-0 overflow-hidden min-h-0">
                {/* Sidebar Catalog */}
                <SkillCatalog
                    workspaceRoot={workspaceRoot}
                    onFileSelect={loadFile}
                    currentFilePath={currentPath}
                />

                {/* Main Editor Area */}
                <div className="flex-1 flex flex-col min-w-0 bg-base-200/50">
                    {/* Header / Toolbar */}
                    <div className="flex items-center justify-between px-6 py-3 bg-base-100/30 border-b border-base-content/5">
                        <div className="flex items-center gap-3 min-w-0">
                            <div className="min-w-0">
                                <h2 className="text-sm font-bold truncate flex items-center gap-2">
                                    {currentPath ? currentPath.split(/[\\/]/).pop() : "Editor"}
                                    {content && (
                                        <span className="text-[11px] font-mono opacity-50 font-normal px-2 py-0.5 bg-base-300 rounded border border-base-content/5">
                                            {content.length} chars | ({Math.ceil(content.length / 4)} est tokens)
                                        </span>
                                    )}
                                </h2>
                                <p className="text-[11px] opacity-50 truncate tooltip tooltip-bottom" data-tip={currentPath}>
                                    {currentPath || "Select a file to start editing"}
                                </p>
                            </div>
                            {isDirty && (
                                <div className="badge badge-warning badge-xs gap-1 py-2 px-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-warning-content animate-pulse" />
                                    <span>Unsaved</span>
                                </div>
                            )}
                        </div>

                        <div className="flex items-center gap-2">
                            {isDirty && (
                                <button
                                    className="btn btn-ghost btn-sm gap-2 opacity-60 hover:opacity-100"
                                    onClick={handleDiscard}
                                    disabled={saving}
                                >
                                    <Icon icon="lucide:undo-2" className="text-lg" />
                                    Discard
                                </button>
                            )}
                            <button
                                className={`btn btn-primary btn-sm gap-2 ${saving ? "loading" : ""}`}
                                disabled={!isDirty || saving}
                                onClick={handleSave}
                            >
                                <Icon icon="lucide:save" className="text-lg" />
                                Save
                            </button>
                        </div>
                    </div>

                    {/* Editor Surface */}
                    <div className="flex-1 flex flex-col min-h-0 overflow-hidden bg-base-200/20 rounded-b-2xl">
                        {isLoading ? (
                            <div className="flex-1 flex flex-col items-center justify-center opacity-40">
                                <span className="loading loading-spinner loading-lg text-primary"></span>
                                <span className="text-xs mt-4 font-bold tracking-widest uppercase animate-pulse">Initializing...</span>
                            </div>
                        ) : currentPath ? (
                            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                                <MarkdownEditor
                                    key={currentPath.toLowerCase()}
                                    markdown={content}
                                    onChange={handleContentChange}
                                    theme={theme}
                                />
                            </div>
                        ) : (
                            <div className="flex-1 flex flex-col items-center justify-center opacity-30 text-center p-6">
                                <div className="p-8 bg-base-300 rounded-full mb-6 shadow-inner ring-1 ring-base-content/5">
                                    <Icon icon="lucide:brain-circuit" className="text-7xl text-primary/50" />
                                </div>
                                <h2 className="text-2xl font-bold tracking-tight text-base-content/80">Behavior Adjustment</h2>
                                <p className="max-w-xs mx-auto mt-3 text-sm leading-relaxed">Select an AGENTS.md or SKILL.md from the sidebar to refine your AI's behavioral rules.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Discard Changes Modal */}
            {showConfirmModal && (
                <div className="modal modal-open">
                    <div className="modal-box shadow-2xl border border-base-300">
                        <h3 className="font-bold text-lg flex items-center gap-2">
                            <Icon icon="lucide:alert-triangle" className="text-warning" />
                            Unsaved Changes
                        </h3>
                        <p className="py-4">You have unsaved changes in your current file. Do you want to discard them and switch to the new file?</p>
                        <div className="modal-action">
                            <button className="btn btn-ghost" onClick={() => setShowConfirmModal(false)}>Cancel</button>
                            <button className="btn btn-error" onClick={confirmDiscardModal}>Discard Changes</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}


