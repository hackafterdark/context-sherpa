import { useEffect, useState, useRef } from 'react';
import { Icon } from '@iconify/react';
import { GetWorkspaces, OpenWorkspace, PickDirectory, RegisterWorkspace, RunIndexingTask, FocusWorkspaceClient } from '../wailsjs/go/main/App';
import { EventsOn } from '../wailsjs/runtime/runtime';

type Workspace = {
    pid: number;
    root: string;
    client: string;
    state: string;
};

type IndexingStatus = {
    root: string;
    logs: string[];
    isActive: boolean;
};

export default function Home({ onVisualize }: { onVisualize: (root: string) => void }) {
    const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
    const [indexing, setIndexing] = useState<IndexingStatus | null>(null);
    const [focusingPid, setFocusingPid] = useState<number | null>(null);
    const logEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchWorkspaces = async () => {
            try {
                const ws = await GetWorkspaces();
                setWorkspaces(ws as Workspace[]);
            } catch (e) {
                console.error("Error fetching workspaces:", e);
            }
        };

        fetchWorkspaces();

        // Listen for real-time updates from the Hub
        const offUpdate = EventsOn('workspace-updated', (ws: Workspace[]) => {
            setWorkspaces(ws);
        });

        // Listen for indexing logs
        const offLog = EventsOn('indexing-log', (data: { root: string, message: string }) => {
            setIndexing(prev => {
                if (!prev || prev.root !== data.root) return prev;
                return { ...prev, logs: [...prev.logs, data.message] };
            });
        });

        const offFinish = EventsOn('indexing-finished', (data: { root: string, success: boolean, error?: string }) => {
            setIndexing(prev => {
                if (!prev || prev.root !== data.root) return prev;
                return { ...prev, isActive: false, logs: [...prev.logs, data.success ? "--- Indexing complete! ---" : `--- Error: ${data.error} ---`] };
            });
        });

        const interval = setInterval(fetchWorkspaces, 10000); // Slower polling as backup

        return () => {
            clearInterval(interval);
            offUpdate();
            offLog();
            offFinish();
        };
    }, []);

    useEffect(() => {
        if (logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [indexing?.logs]);

    const handleAddWorkspace = async () => {
        try {
            const path = await PickDirectory();
            if (path) {
                await RegisterWorkspace(path);
            }
        } catch (e) {
            console.error("Error adding workspace:", e);
        }
    };

    const handleStartIndexing = async (root: string) => {
        setIndexing({ root, logs: ["Starting indexing task..."], isActive: true });
        try {
            await RunIndexingTask(root);
        } catch (e) {
            console.error("Error starting indexing:", e);
        }
    };

    const handleFocusClient = async (pid: number) => {
        if (!pid) return;
        setFocusingPid(pid);
        try {
            await FocusWorkspaceClient(pid);
        } catch (e) {
            console.error("Error focusing client:", e);
        } finally {
            setTimeout(() => setFocusingPid(null), 1000);
        }
    };

    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-300 max-w-5xl">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold font-sans tracking-tight">Dashboard</h1>
                <button 
                    onClick={handleAddWorkspace}
                    className="btn btn-primary btn-sm gap-2"
                >
                    <Icon icon="lucide:plus" />
                    Add Workspace
                </button>
            </div>

            <div className="flex flex-col gap-6">
                <div className="card bg-base-100 shadow-sm border border-base-200">
                    <div className="card-body">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="card-title text-xl flex items-center gap-2">
                                <Icon icon="lucide:folder-tree" className="text-primary" />
                                Workspaces
                            </h2>
                        </div>

                        {workspaces.length === 0 ? (
                            <p className="text-base-content/70 flex items-center gap-2 italic">
                                <Icon icon="lucide:info" />
                                No workspaces registered. Add a workspace or start an MCP node.
                            </p>
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="table table-sm w-full">
                                    <thead>
                                        <tr>
                                            <th>Path</th>
                                            <th>Client</th>
                                            <th>Status</th>
                                            <th>PID</th>
                                            <th className="text-right">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {workspaces.map((ws) => (
                                            <tr key={`${ws.root}-${ws.pid}`} className={`hover group ${ws.state === 'offline' ? 'opacity-70' : ''}`}>
                                                <td className="font-mono text-xs max-w-xs truncate" title={ws.root}>
                                                    {ws.root}
                                                </td>
                                                <td>
                                                    <button 
                                                        onClick={() => handleFocusClient(ws.pid)}
                                                        disabled={!ws.pid || ws.state === 'offline'}
                                                        className={`badge badge-sm cursor-pointer transition-all duration-200 border-none hover:brightness-90 active:scale-95 ${
                                                            focusingPid === ws.pid 
                                                                ? 'badge-primary animate-pulse' 
                                                                : 'badge-ghost'
                                                        }`}
                                                        title={ws.pid ? "Focus Editor" : "No process information"}
                                                    >
                                                        {focusingPid === ws.pid ? 'Focusing...' : ws.client}
                                                    </button>
                                                </td>
                                                <td>
                                                    <span className={`badge badge-sm ${ws.state === 'active' ? 'badge-success' :
                                                        ws.state === 'offline' ? 'badge-ghost' :
                                                            'badge-warning'
                                                        }`}>
                                                        {ws.state}
                                                    </span>
                                                </td>
                                                <td className="opacity-50 text-xs">{ws.pid || '-'}</td>
                                                <td className="text-right whitespace-nowrap">
                                                    <div className="flex justify-end gap-1">
                                                        <button
                                                            onClick={() => handleStartIndexing(ws.root)}
                                                            className="btn btn-ghost btn-xs btn-square tooltip tooltip-left text-primary hover:bg-primary/10"
                                                            data-tip="Build SCIP Index"
                                                        >
                                                            <Icon icon="lucide:refresh-cw" className={`w-4 h-4 ${indexing?.root === ws.root && indexing.isActive ? 'animate-spin' : ''}`} />
                                                        </button>
                                                        <button
                                                            onClick={() => onVisualize(ws.root)}
                                                            className="btn btn-ghost btn-xs btn-square tooltip tooltip-left text-accent hover:bg-accent/10"
                                                            data-tip="Visualize Code Atlas"
                                                        >
                                                            <Icon icon="lucide:network" className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => OpenWorkspace(ws.root)}
                                                            className="btn btn-ghost btn-xs btn-square tooltip tooltip-left text-primary hover:bg-primary/10"
                                                            data-tip="Open in Explorer"
                                                        >
                                                            <Icon icon="lucide:folder-open" className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Indexing Modal/Overlay */}
            {indexing && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-in fade-in duration-200">
                    <div className="card w-full max-w-2xl bg-base-100 shadow-2xl border border-base-300">
                        <div className="card-body p-6">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-bold flex items-center gap-2">
                                    <Icon icon="lucide:cpu" className="text-primary" />
                                    Indexing: {indexing.root.split(/[\\/]/).pop()}
                                </h3>
                                {!indexing.isActive && (
                                    <button 
                                        onClick={() => setIndexing(null)}
                                        className="btn btn-ghost btn-sm btn-circle"
                                    >
                                        <Icon icon="lucide:x" />
                                    </button>
                                )}
                            </div>

                            <div className="bg-black/90 rounded-lg p-4 h-64 overflow-y-auto font-mono text-xs text-green-400 border border-base-content/10">
                                {indexing.logs.map((log, i) => (
                                    <div key={i} className="mb-1">{log}</div>
                                ))}
                                <div ref={logEndRef} />
                            </div>

                            <div className="mt-4 flex justify-between items-center">
                                <div className="flex items-center gap-3">
                                    {indexing.isActive ? (
                                        <>
                                            <span className="loading loading-spinner loading-sm text-primary"></span>
                                            <span className="text-sm font-medium animate-pulse">Running scip indexer...</span>
                                        </>
                                    ) : (
                                        <span className="text-sm font-medium text-success flex items-center gap-2">
                                            <Icon icon="lucide:check-circle" />
                                            Task Finished
                                        </span>
                                    )}
                                </div>
                                {!indexing.isActive && (
                                    <div className="flex gap-2">
                                        <button 
                                            onClick={() => {
                                                setIndexing(null);
                                                onVisualize(indexing.root);
                                            }}
                                            className="btn btn-accent btn-sm gap-2"
                                        >
                                            <Icon icon="lucide:network" />
                                            View in Atlas
                                        </button>
                                        <button 
                                            onClick={() => setIndexing(null)}
                                            className="btn btn-ghost btn-sm"
                                        >
                                            Close
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
