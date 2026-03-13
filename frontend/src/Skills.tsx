import { useState, useEffect } from 'react';
import { Icon } from '@iconify/react';
import { ReadMarkdown, WriteMarkdown, GetWorkspaces } from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';
import MarkdownEditor from './components/MarkdownEditor';
import SkillCatalog from './components/SkillCatalog';

type SkillsProps = {
    workspaceRoot: string;
    onWorkspaceChange: (root: string) => void;
};

export default function Skills({ workspaceRoot, onWorkspaceChange }: SkillsProps) {
    const [currentPath, setCurrentPath] = useState('');
    const [content, setContent] = useState('');
    const [originalContent, setOriginalContent] = useState('');
    const [isDirty, setIsDirty] = useState(false);
    const [saving, setSaving] = useState(false);
    const [pendingPath, setPendingPath] = useState<string | null>(null);
    const [showConfirmModal, setShowConfirmModal] = useState(false);
    const [allWorkspaces, setAllWorkspaces] = useState<any[]>([]);

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

    const loadFile = async (path: string) => {
        if (isDirty) {
            setPendingPath(path);
            setShowConfirmModal(true);
            return;
        }

        try {
            const data = await ReadMarkdown(path);
            setCurrentPath(path);
            setContent(data);
            setOriginalContent(data);
            setIsDirty(false);
        } catch (err) {
            console.error("Failed to load markdown:", err);
        }
    };

    const handleContentChange = (newContent: string) => {
        setContent(newContent);
        setIsDirty(newContent !== originalContent);
    };

    const handleSave = async () => {
        if (!currentPath || !isDirty) return;

        setSaving(true);
        try {
            await WriteMarkdown(currentPath, content);
            setOriginalContent(content);
            setIsDirty(false);
        } catch (err) {
            console.error("Failed to save markdown:", err);
        } finally {
            setSaving(false);
        }
    };

    const confirmDiscard = () => {
        setIsDirty(false);
        setShowConfirmModal(false);
        if (pendingPath) {
            loadFile(pendingPath);
            setPendingPath(null);
        }
    };

    return (
        <div className="flex flex-col flex-1 h-full w-full gap-4 animate-in fade-in duration-300 bg-base-100">
            {/* Main Header matching Atlas.tsx */}
            <div className="flex justify-between items-center px-6 pt-5 shrink-0">
                <div className="flex items-center gap-4">
                    <h1 className="text-3xl font-bold font-sans tracking-tight">Intelligence Command</h1>
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
                                            <span className="text-[9px] opacity-30 font-mono truncate w-full pl-6 leading-none">
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
                            <div className="p-2 bg-primary/10 rounded-lg">
                                <Icon icon="lucide:file-edit" className="text-primary text-xl" />
                            </div>
                            <div className="min-w-0">
                                <h2 className="text-sm font-bold truncate">
                                    {currentPath ? currentPath.split(/[\\/]/).pop() : "Editor"}
                                </h2>
                                <p className="text-[10px] opacity-50 truncate cursor-help tooltip tooltip-bottom" data-tip={currentPath}>
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
                            <button 
                                className={`btn btn-primary btn-sm gap-2 ${saving ? "loading" : ""}`}
                                disabled={!isDirty || saving}
                                onClick={handleSave}
                            >
                                <Icon icon="lucide:save" className="text-lg" />
                                Save to Disk
                            </button>
                        </div>
                    </div>

                    {/* Editor Surface */}
                    <div className="flex-1 p-6 overflow-hidden">
                        {currentPath ? (
                            <MarkdownEditor 
                                markdown={content}
                                onChange={handleContentChange}
                            />
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center opacity-30 text-center">
                                <Icon icon="lucide:brain-circuit" className="text-8xl mb-4" />
                                <h2 className="text-2xl font-bold">Refine Intelligence</h2>
                                <p>Select an AGENTS.md or SKILL.md from the sidebar to refine your AI's mind.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Discard Changes Modal */}
            {showConfirmModal && (
                <div className="modal modal-open">
                    <div className="modal-box">
                        <h3 className="font-bold text-lg flex items-center gap-2">
                            <Icon icon="lucide:alert-triangle" className="text-warning" />
                            Unsaved Changes
                        </h3>
                        <p className="py-4">You have unsaved changes in your current file. Do you want to discard them and switch to the new file?</p>
                        <div className="modal-action">
                            <button className="btn btn-ghost" onClick={() => setShowConfirmModal(false)}>Cancel</button>
                            <button className="btn btn-error" onClick={confirmDiscard}>Discard Changes</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

