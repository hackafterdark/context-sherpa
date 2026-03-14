import { useState, useEffect } from 'react';
import { Icon } from '@iconify/react';
import { DiscoverMarkdownFiles } from '../../wailsjs/go/main/App';

type MarkdownFile = {
    path: string;
    name: string;
    frontMatter?: Record<string, string>;
};

type SkillCatalogProps = {
    workspaceRoot: string;
    onFileSelect: (path: string) => void;
    currentFilePath: string;
};

export default function SkillCatalog({ workspaceRoot, onFileSelect, currentFilePath }: SkillCatalogProps) {
    const [files, setFiles] = useState<MarkdownFile[]>([]);
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");

    useEffect(() => {
        if (workspaceRoot) {
            refreshCatalog();
        }
    }, [workspaceRoot]);

    const refreshCatalog = async () => {
        setLoading(true);
        try {
            const rawResults = await DiscoverMarkdownFiles(workspaceRoot);

            // Strictly filter for SKILL.md and AGENTS.md (recursive)
            const filteredResults = rawResults.filter((entry: any) => {
                const base = entry.path.split(/[\\/]/).pop()?.toUpperCase();
                return base === 'SKILL.MD' || base === 'AGENTS.MD';
            });

            const discoveredFiles: MarkdownFile[] = filteredResults.map((entry: any) => {
                const relative = entry.path.startsWith(workspaceRoot)
                    ? entry.path.slice(workspaceRoot.length).replace(/^[\\/]/, '')
                    : entry.path;

                return {
                    path: entry.path,
                    name: relative,
                    frontMatter: entry.frontMatter
                };
            });
            setFiles(discoveredFiles);
        } catch (err) {
            console.error("Failed to load skill catalog:", err);
        } finally {
            setLoading(false);
        }
    };

    const filteredFiles = files.filter(file => {
        const query = searchTerm.toLowerCase();
        const inName = file.name.toLowerCase().includes(query);
        const inFrontMatterName = file.frontMatter?.name?.toLowerCase().includes(query);
        const inFrontMatterDesc = file.frontMatter?.description?.toLowerCase().includes(query);
        return inName || inFrontMatterName || inFrontMatterDesc;
    });

    return (
        <div className="w-80 bg-base-300 flex flex-col h-full border-r border-base-content/10">
            <div className="p-4 border-b border-base-content/5">
                <div className="relative">
                    <Icon icon="lucide:search" className="absolute left-3 top-1/2 -translate-y-1/2 text-base-content/40" />
                    <input
                        type="text"
                        placeholder="Search rules or skills..."
                        className="input input-sm input-bordered w-full pl-10 bg-base-200 focus:outline-none"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                    <button
                        className="absolute right-2 top-1/2 -translate-y-1/2 btn btn-ghost btn-xs btn-square opacity-40 hover:opacity-100"
                        onClick={refreshCatalog}
                        disabled={loading}
                    >
                        <Icon icon="lucide:rotate-cw" className={loading ? "animate-spin" : ""} />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-hidden overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-base-content/10">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-40 gap-2 opacity-50">
                        <span className="loading loading-spinner loading-md"></span>
                        <span className="text-xs">Scanning...</span>
                    </div>
                ) : filteredFiles.length === 0 ? (
                    <div className="p-8 text-center opacity-50">
                        <Icon icon="lucide:file-question" className="text-3xl mx-auto mb-2" />
                        <p className="text-sm">No AGENTS.md or SKILL.md found.</p>
                    </div>
                ) : (
                    <div className="space-y-1">
                        {filteredFiles.map(file => {
                            const fileName = file.path.split(/[\\/]/).pop() || '';
                            const dirName = file.name.substring(0, file.name.lastIndexOf(fileName)).replace(/[\\/]$/, '');
                            const isActive = file.path.toLowerCase() === currentFilePath.toLowerCase();

                            return (
                                <button
                                    key={file.path}
                                    onClick={() => onFileSelect(file.path)}
                                    className={`w-full text-left group/item tooltip tooltip-right before:text-[10px] before:max-w-[300px] before:content-[attr(data-tip)] p-3 rounded-lg flex items-center gap-3 transition-all ${isActive ? "bg-primary/10 text-primary ring-1 ring-primary/30 shadow-lg shadow-primary/5" : "hover:bg-base-200 opacity-60 hover:opacity-100"}`}
                                    data-tip={file.path}
                                >
                                    <div className={`p-2 rounded-md ${isActive ? "bg-primary/20" : "bg-base-200"}`}>
                                        <Icon
                                            icon={fileName.toUpperCase() === 'AGENTS.MD' ? "lucide:shield-check" : "lucide:zap"}
                                            className={`shrink-0 ${isActive ? "text-primary" : "text-base-content"}`}
                                        />
                                    </div>
                                    <div className="min-w-0 flex-1">
                                        <div className="truncate text-[13px] font-bold">{fileName}</div>
                                        {(file.frontMatter?.name || dirName) && (
                                            <div className="text-xs opacity-50 truncate font-normal leading-tight mt-0.5">
                                                {file.frontMatter?.name || dirName}
                                            </div>
                                        )}
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}


