import { useEffect, useState } from 'react';
import { Icon } from '@iconify/react';
import { GetWorkspaces, OpenWorkspace } from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';

type Workspace = {
    pid: number;
    root: string;
    client: string;
    state: string;
};

export default function Home({ onVisualize }: { onVisualize: (root: string) => void }) {
    const [workspaces, setWorkspaces] = useState<Workspace[]>([]);

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
        EventsOn('workspace-updated', (ws: Workspace[]) => {
            setWorkspaces(ws);
        });

        const interval = setInterval(fetchWorkspaces, 10000); // Slower polling as backup

        return () => {
            clearInterval(interval);
            EventsOff('workspace-updated');
        };
    }, []);

    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-300">
            <h1 className="text-3xl font-bold font-sans tracking-tight">Dashboard</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="card bg-base-100 shadow-sm border border-base-200 col-span-1 md:col-span-2">
                    <div className="card-body">
                        <h2 className="card-title text-xl mb-4 flex items-center gap-2">
                            <Icon icon="lucide:folder-tree" className="text-primary" />
                            Active Workspaces
                        </h2>

                        {workspaces.length === 0 ? (
                            <p className="text-base-content/70 flex items-center gap-2 italic">
                                <Icon icon="lucide:info" />
                                No active workspaces detected. Start an MCP node to register a workspace.
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
                                            <tr key={`${ws.root}-${ws.pid}`} className={`hover group ${ws.state === 'offline' ? 'opacity-50 grayscale' : ''}`}>
                                                <td className="font-mono text-xs max-w-xs truncate" title={ws.root}>{ws.root}</td>
                                                <td><span className="badge badge-ghost badge-sm">{ws.client}</span></td>
                                                <td>
                                                    <span className={`badge badge-sm ${ws.state === 'active' ? 'badge-success' :
                                                        ws.state === 'offline' ? 'badge-ghost' :
                                                            'badge-warning'
                                                        }`}>
                                                        {ws.state}
                                                    </span>
                                                </td>
                                                <td className="opacity-50 text-xs">{ws.pid}</td>
                                                <td className="text-right whitespace-nowrap">
                                                    <div className="flex justify-end gap-1">
                                                        <button
                                                            onClick={() => onVisualize(ws.root)}
                                                            className="btn btn-ghost btn-xs btn-square tooltip tooltip-left text-accent hover:bg-accent/10"
                                                            data-tip="Visualize Codebase"
                                                        >
                                                            <Icon icon="lucide:network" className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => OpenWorkspace(ws.root)}
                                                            className="btn btn-ghost btn-xs btn-square tooltip tooltip-left text-primary hover:bg-primary/10"
                                                            data-tip="Browse Folder"
                                                        >
                                                            <Icon icon="lucide:external-link" className="w-4 h-4" />
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

                {/* <div className="card bg-base-100 shadow-sm border border-base-200">
                    <div className="card-body">
                        <h2 className="card-title text-lg flex items-center gap-2">
                            <Icon icon="lucide:database" className="text-primary" />
                            Local Memory
                        </h2>
                        <p className="text-base-content/70 text-sm">
                            `sherpa.db` state tracking and task history are stored within each workspace root.
                        </p>
                    </div>
                </div> */}
            </div>
        </div>
    );
}
