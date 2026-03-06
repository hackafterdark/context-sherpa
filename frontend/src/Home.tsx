import { useEffect, useState } from 'react';
import { Icon } from '@iconify/react';
import { GetWorkspaces } from '../wailsjs/go/main/App';

type Workspace = {
    pid: number;
    root: string;
    client: string;
    state: string;
};

export default function Home() {
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
        const interval = setInterval(fetchWorkspaces, 3000);
        return () => clearInterval(interval);
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
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {workspaces.map((ws, i) => (
                                            <tr key={i} className="hover">
                                                <td className="font-mono text-xs">{ws.root}</td>
                                                <td><span className="badge badge-ghost badge-sm">{ws.client}</span></td>
                                                <td><span className={`badge badge-sm ${ws.state === 'active' ? 'badge-success' : 'badge-warning'}`}>{ws.state}</span></td>
                                                <td className="opacity-50 text-xs">{ws.pid}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>

                <div className="card bg-base-100 shadow-sm border border-base-200">
                    <div className="card-body">
                        <h2 className="card-title text-lg flex items-center gap-2">
                            <Icon icon="lucide:database" className="text-primary" />
                            Local Memory
                        </h2>
                        <p className="text-base-content/70 text-sm">
                            `sherpa.db` state tracking and task history are stored within each workspace root.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
