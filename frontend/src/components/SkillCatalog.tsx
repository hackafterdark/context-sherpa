import { useState, useEffect } from 'react';
import { Icon } from '@iconify/react';
import { DiscoverMarkdownFiles, ReadMarkdown } from '../../wailsjs/go/main/App';

type Skill = {
    name: string;
    line: number;
    filePath: string;
};

type MarkdownFile = {
    path: string;
    name: string;
    skills: Skill[];
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
            const rawFilePaths = await DiscoverMarkdownFiles(workspaceRoot);
            
            // Strictly filter for SKILL.md and AGENTS.md (recursive)
            const filePaths = rawFilePaths.filter((p: string) => {
                const base = p.split(/[\\/]/).pop()?.toUpperCase();
                return base === 'SKILL.MD' || base === 'AGENTS.MD';
            });

            const discoveredFiles: MarkdownFile[] = [];

            for (const path of filePaths) {
                const content = await ReadMarkdown(path);
                const skills: Skill[] = [];
                const lines = content.split('\n');
                
                lines.forEach((line: string, index: number) => {
                    const match = line.match(/^#+\s+Skill:\s*(.*)/i) || line.match(/^#+\s+(.*)/);
                    if (match) {
                        skills.push({
                            name: match[1].trim(),
                            line: index + 1,
                            filePath: path
                        });
                    }
                });

                // Calculate a nice display name showing directory context
                const relative = path.startsWith(workspaceRoot) 
                    ? path.slice(workspaceRoot.length).replace(/^[\\/]/, '') 
                    : path;
                
                discoveredFiles.push({
                    path,
                    name: relative, // Use relative path for distinction
                    skills
                });
            }
            setFiles(discoveredFiles);
        } catch (err) {
            console.error("Failed to load skill catalog:", err);
        } finally {
            setLoading(false);
        }
    };

    const filteredFiles = files.filter(file => 
        file.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        file.skills.some(skill => skill.name.toLowerCase().includes(searchTerm.toLowerCase()))
    );

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

            <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-base-content/10">
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
                            
                            return (
                                <div key={file.path} className="collapse collapse-arrow bg-base-100/30 rounded-lg group/item tooltip tooltip-right before:text-[10px] before:max-w-[300px] before:content-[attr(data-tip)]" data-tip={file.path}>
                                    <input type="checkbox" defaultChecked={file.path === currentFilePath} /> 
                                    <div 
                                        className={`collapse-title text-sm font-semibold flex items-center gap-2 p-3 min-h-0 ${file.path === currentFilePath ? "text-primary bg-primary/10 rounded-t-lg" : ""}`}
                                        onClick={() => onFileSelect(file.path)}
                                    >
                                        <Icon icon={fileName.toUpperCase() === 'AGENTS.MD' ? "lucide:shield-check" : "lucide:zap"} className="shrink-0 text-primary/70" />
                                        <div className="min-w-0 flex-1">
                                            <div className="truncate text-xs font-bold">{fileName}</div>
                                            {dirName && (
                                                <div className="text-[11px] opacity-50 truncate font-normal leading-tight">
                                                    {dirName}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                    <div className="collapse-content p-0 pl-1 pr-1">
                                        <ul className="menu menu-xs p-1">
                                            {file.skills.map((skill, i) => (
                                                <li key={i}>
                                                    <a 
                                                        className="flex items-center gap-2 py-1.5 hover:bg-base-content/5 rounded-md"
                                                        onClick={() => onFileSelect(file.path)}
                                                    >
                                                        <Icon icon="lucide:hash" className="text-[10px] opacity-30" />
                                                        <span className="truncate">{skill.name}</span>
                                                    </a>
                                                </li>
                                            ))}
                                            {file.skills.length === 0 && (
                                                <li className="disabled text-xs opacity-50 px-3 py-2 italic text-center">No sections</li>
                                            )}
                                        </ul>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}


