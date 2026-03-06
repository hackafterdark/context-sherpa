import { Icon } from '@iconify/react';

export default function Home() {
    return (
        <div className="flex flex-col gap-6 animate-in fade-in duration-300">
            <h1 className="text-3xl font-bold font-sans tracking-tight">Dashboard</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="card bg-base-100 shadow-sm border border-base-200">
                    <div className="card-body">
                        <h2 className="card-title text-lg">Active Project</h2>
                        <p className="text-base-content/70 flex items-center gap-2">
                            <Icon icon="lucide:folder-open" />
                            No project is currently loaded.
                        </p>
                    </div>
                </div>

                <div className="card bg-base-100 shadow-sm border border-base-200">
                    <div className="card-body">
                        <h2 className="card-title text-lg">Local Index</h2>
                        <p className="text-base-content/70 flex items-center gap-2">
                            <Icon icon="lucide:database" />
                            Vectors and SCIP metrics will appear here.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
