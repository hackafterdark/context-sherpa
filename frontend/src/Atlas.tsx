import { useEffect, useState, useRef } from 'react';
import { Icon } from '@iconify/react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { SearchForIndexes, GetGraphData, GetWorkspaces, GetFileContent, RegenerateIndex, SearchSymbols, GetSymbolRelationships } from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

cytoscape.use(fcose);

type AtlasProps = {
    workspaceRoot: string;
    onWorkspaceChange: (root: string) => void;
};

export default function Atlas({ workspaceRoot, onWorkspaceChange }: AtlasProps) {
    const chartRef = useRef<HTMLDivElement>(null);
    const cyRef = useRef<cytoscape.Core | null>(null);
    const [indexFiles, setIndexFiles] = useState<string[]>([]);
    const [selectedIndex, setSelectedIndex] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedNode, setSelectedNode] = useState<any>(null);
    const [selectedEdge, setSelectedEdge] = useState<any>(null);
    const [language, setLanguage] = useState<string>('Go');
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [showSearchResults, setShowSearchResults] = useState(false);
    const [allWorkspaces, setAllWorkspaces] = useState<any[]>([]);
    const searchRef = useRef<HTMLDivElement>(null);
    const [fileContent, setFileContent] = useState<string>('');
    const [isSourceExpanded, setIsSourceExpanded] = useState(false);
    const editorRef = useRef<any>(null);
    const [regenerating, setRegenerating] = useState<Record<string, boolean>>({});
    const [indexingStatus, setIndexingStatus] = useState<{ type: 'success' | 'error' | 'processing', message: string } | null>(null);
    const [currentScope, setCurrentScope] = useState('');
    const [relationships, setRelationships] = useState<{ callers: any[], callees: any[] } | null>(null);
    const [pendingFocus, setPendingFocus] = useState<string | null>(null);
    const [loadingRelationships, setLoadingRelationships] = useState(false);
    const [graphData, setGraphData] = useState<any>(null);

    const getKindLabel = (kind: string, plural = false) => {
        if (kind === 'Struct') {
            const isClass = ['TypeScript', 'JavaScript', 'Python'].includes(language);
            if (isClass) return plural ? 'Classes' : 'Class';
            return plural ? 'Structs' : 'Struct';
        }
        if (plural) return kind + 's';
        return kind;
    };

    const buildTree = (files: string[], root: string) => {
        const tree: any[] = [];
        files.forEach(file => {
            const relative = file.startsWith(root) ? file.slice(root.length).replace(/^[\\/]/, '') : file;
            const parts = relative.split(/[\\/]/);
            let current = tree;
            let currentPath = '';

            parts.forEach((part, i) => {
                const isLast = i === parts.length - 1;
                currentPath = currentPath ? `${currentPath}/${part}` : part;
                let node = current.find(n => n.name === part);
                if (!node) {
                    node = {
                        name: part,
                        path: isLast ? file : currentPath,
                        kind: isLast ? 'file' : 'folder',
                        children: []
                    };
                    current.push(node);
                }
                current = node.children;
            });
        });
        return tree;
    };

    const handleRegenerateIndex = async (path: string) => {
        setRegenerating(prev => ({ ...prev, [path]: true }));
        setIndexingStatus(null);
        try {
            await RegenerateIndex(workspaceRoot, path);
            setIndexingStatus({ type: 'success', message: 'Index regenerated successfully!' });
            // Re-fetch indexes to ensure everything is up to date
            const files = await SearchForIndexes(workspaceRoot);
            setIndexFiles(files || []);
        } catch (e: any) {
            console.error("Error regenerating index:", e);
            const msg = e?.toString() || 'Unknown error';
            setIndexingStatus({
                type: 'error',
                message: msg.includes('missing tsconfig.json')
                    ? 'Failed: Missing tsconfig.json in project root.'
                    : `Indexing failed: ${msg.split('\n')[0]}`
            });
        } finally {
            setRegenerating(prev => ({ ...prev, [path]: false }));
        }
    };

    const renderTree = (nodes: any[], depth = 0) => {
        return nodes.sort((a, b) => (a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === 'folder' ? -1 : 1)).map(node => (
            <div key={node.path}>
                {node.kind === 'folder' ? (
                    <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-black opacity-30 uppercase tracking-[0.2em] select-none mt-2 first:mt-0">
                        <div style={{ width: depth * 12 }} />
                        <Icon icon="lucide:folder" className="w-2.5 h-2.5" />
                        {node.name}
                    </div>
                ) : (
                    <div className={`group flex items-center pr-2 rounded-lg transition-all border ${selectedIndex === node.path
                        ? 'bg-primary/10 text-primary font-bold border-primary/20'
                        : 'hover:bg-base-200 border-transparent'
                        }`}
                        style={{ marginLeft: depth * 12 }}
                    >
                        <button
                            key={node.path}
                            onClick={() => setSelectedIndex(node.path)}
                            className="flex items-center gap-2 flex-1 px-3 py-1.5 text-[11px] text-left min-w-0"
                        >
                            <Icon icon="lucide:file-code" className={`w-3 h-3 flex-shrink-0 ${selectedIndex === node.path ? 'opacity-100' : 'opacity-30 group-hover:opacity-100'}`} />
                            <span className={`truncate ${selectedIndex === node.path ? 'opacity-100' : 'opacity-60 group-hover:opacity-100'}`}>{node.name}</span>
                        </button>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                handleRegenerateIndex(node.path);
                            }}
                            disabled={regenerating[node.path]}
                            className={`p-1 rounded-md transition-all ${regenerating[node.path] || selectedIndex === node.path
                                ? 'opacity-100'
                                : 'opacity-0 group-hover:opacity-60 hover:opacity-100 hover:bg-base-300'
                                }`}
                            title="Regenerate Index"
                        >
                            <Icon
                                icon={regenerating[node.path] ? "lucide:loader-2" : "lucide:refresh-cw"}
                                className={`w-3 h-3 ${regenerating[node.path] ? 'animate-spin text-primary' : ''} ${selectedIndex === node.path && !regenerating[node.path] ? 'text-primary' : ''}`}
                            />
                        </button>
                    </div>
                )}
                {node.children && node.children.length > 0 && renderTree(node.children, depth + 1)}
            </div>
        ));
    };

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
                setShowSearchResults(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        if (indexingStatus && indexingStatus.type !== 'processing') {
            const timer = setTimeout(() => setIndexingStatus(null), 5000);
            return () => clearTimeout(timer);
        }
    }, [indexingStatus]);

    useEffect(() => {
        const fetchWorkspaces = async () => {
            try {
                const ws = await GetWorkspaces();
                // De-dupe by root path
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

		EventsOn('indexing-status', (data: any) => {
			if (data.status === 'done') {
				setIndexingStatus({ type: 'success', message: data.message || 'Indexing complete!' });
			} else if (data.status === 'error') {
				setIndexingStatus({ type: 'error', message: data.message || 'Indexing failed.' });
			} else {
				setIndexingStatus({
					type: 'processing',
					message: data.message || 'Indexing...'
				});
			}
		});

		return () => {
			EventsOff('workspace-updated');
			EventsOff('indexing-status');
		};
    }, []);

    useEffect(() => {
        if (workspaceRoot) {
            setLoading(true); // Add loading state when switching
            SearchForIndexes(workspaceRoot).then((files) => {
                setIndexFiles(files || []);
                setLoading(false);
                if (files && files.length > 0) {
                    setSelectedIndex(files[0]);
                }
            });
        }
    }, [workspaceRoot]);

    useEffect(() => {
        if (chartRef.current && !cyRef.current) {
            cyRef.current = cytoscape({
                container: chartRef.current,
                elements: [],
                style: [
                    {
                        selector: 'node',
                        style: {
                            'width': 'data(value)',
                            'height': 'data(value)',
                            'background-color': (node: any) => {
                                const kind = node.data('kind');
                                return kind === 'Struct' ? '#f59e0b' :
                                    kind === 'Interface' ? '#10b981' :
                                        kind === 'Function' ? '#3b82f6' :
                                            kind === 'Variable' ? '#6b7280' : '#222';
                            },
                            'label': 'data(name)',
                            'font-size': '8px',
                            'color': 'rgba(255,255,255,0.6)',
                            'text-valign': 'bottom',
                            'text-halign': 'center',
                            'text-margin-y': 4,
                            'min-zoomed-font-size': 10, // Lod optimization: hide when small
                            'z-index': 10,
                            'transition-property': 'background-color, line-color, target-arrow-color, opacity, border-width, border-color',
                            'transition-duration': 200
                        }
                    },
                    {
                        selector: "node[kind = 'Folder']",
                        style: {
                            'background-color': '#4b5563',
                            'background-opacity': 0.1,
                            'border-width': 1,
                            'border-color': '#4b5563',
                            'border-style': 'dashed',
                            'label': 'data(name)',
                            'font-size': '10px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': 'rgba(255,255,255,0.4)',
                            'padding': '40px',
                            'z-index': 1,
                            'shape': 'round-rectangle'
                        }
                    },
                    {
                        selector: 'node:selected',
                        style: {
                            'border-width': 0,
                            'border-color': '#fff',
                            'overlay-color': '#fff',
                            'overlay-padding': 4,
                            'overlay-opacity': 0.2
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 1.5,
                            'line-color': '#333',
                            'target-arrow-color': '#333',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.2,
                            'arrow-scale': 0.8
                        }
                    },
                    {
                        selector: 'node.dimmed, edge.dimmed',
                        style: {
                            'opacity': 0.1,
                            'text-opacity': 0.1
                        }
                    },
                    {
                        selector: 'edge.highlighted',
                        style: {
                            'opacity': 1,
                            'line-color': '#fff',
                            'target-arrow-color': '#fff',
                            'width': 2,
                            'z-index': 19
                        }
                    },
                    {
                        selector: 'node.highlighted',
                        style: {
                            'border-width': 0,
                            'border-color': '#fff',
                            'opacity': 1,
                            'text-opacity': 1,
                            'z-index': 20,
                            'min-zoomed-font-size': 0
                        }
                    }
                ],
                layout: { name: 'null' },
                wheelSensitivity: 1.0,
                boxSelectionEnabled: false,
                autounselectify: false,
                autoungrabify: true
            });

            // Interaction Overhaul - Enable Panning on Nodes
            let isPanning = false;
            let lastPos = { x: 0, y: 0 };

            cyRef.current.on('mousedown', (evt) => {
                isPanning = true;
                lastPos = { x: evt.renderedPosition.x, y: evt.renderedPosition.y };
            });

            cyRef.current.on('mousemove', (evt) => {
                if (isPanning) {
                    const currentPos = { x: evt.renderedPosition.x, y: evt.renderedPosition.y };
                    const dx = currentPos.x - lastPos.x;
                    const dy = currentPos.y - lastPos.y;

                    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
                        cyRef.current?.panBy({ x: dx, y: dy });
                        lastPos = currentPos;
                        // If we are panning, we don't want to trigger a 'tap' (click)
                        cyRef.current?.container()?.classList.add('panning');
                    }
                }
            });

            cyRef.current.on('mouseup', () => {
                isPanning = false;
                setTimeout(() => {
                    cyRef.current?.container()?.classList.remove('panning');
                }, 50);
            });

            cyRef.current.on('tap', 'node', (evt) => {
                // Ignore taps if we were just panning
                if (cyRef.current?.container()?.classList.contains('panning')) return;

                const node = evt.target;
                setSelectedEdge(null);
                setSelectedNode(node.data());
            });

            cyRef.current.on('tap', 'edge', (evt) => {
                const edge = evt.target;
                setSelectedNode(null);
                setSelectedEdge(edge.data());
            });

            cyRef.current.on('tap', (evt) => {
                if (evt.target === cyRef.current) {
                    setSelectedNode(null);
                    setSelectedEdge(null);
                }
            });

            // High-speed hovering
            cyRef.current.on('mouseover', 'node', (evt) => {
                const node = evt.target;
                const nh = node.closedNeighborhood(); // Current node + neighbors + edges
                // Plus edges between neighbors
                const neighborhood = nh.add(nh.edgesWith(nh));

                // Dim EVERYTHING EXCEPT folders and the neighborhood
                cyRef.current?.elements().not('node[kind="Folder"]').addClass('dimmed');
                neighborhood.removeClass('dimmed').addClass('highlighted');
            });

            cyRef.current.on('mouseout', 'node', () => {
                cyRef.current?.elements().removeClass('dimmed').removeClass('highlighted');
            });

            // Double click to drill down into folders
            cyRef.current.on('dblclick', 'node[kind="Folder"]', (evt) => {
                const node = evt.target;
                const folderId = node.id();
                const path = folderId.replace('dir:', '');
                setCurrentScope(path);
            });
        }

        const resizeObserver = new ResizeObserver(() => {
            cyRef.current?.resize();
        });

        if (chartRef.current) {
            resizeObserver.observe(chartRef.current);
        }

        return () => {
            resizeObserver.disconnect();
            cyRef.current?.destroy();
            cyRef.current = null;
        };
    }, []);

    useEffect(() => {
        const node = selectedNode;
        if (node && node.path && workspaceRoot) {
            setRelationships(null);
            GetFileContent(workspaceRoot, node.path).then(content => {
                setFileContent(content);
                if (editorRef.current && node.startLine) {
                    setTimeout(() => {
                        const editor = editorRef.current;
                        editor.revealLineInCenter(node.startLine);
                        (editor as any)._decorations = editor.deltaDecorations((editor as any)._decorations || [], [
                            {
                                range: { startLineNumber: node.startLine, startColumn: 1, endLineNumber: node.endLine || node.startLine, endColumn: 100 },
                                options: {
                                    isWholeLine: true,
                                    className: 'bg-primary/5',
                                    linesDecorationsClassName: 'symbol-gutter-highlight',
                                }
                            }
                        ]);
                    }, 100);
                }
            }).catch(e => {
                console.error("Error fetching file content:", e);
                setFileContent("// Error loading file content");
            });
            setRelationships(null);
            setLoadingRelationships(true);
            GetSymbolRelationships(selectedIndex, node.id).then((rels: any) => {
                setRelationships(rels);
                setLoadingRelationships(false);
            }).catch(() => {
                setLoadingRelationships(false);
            });
        } else {
            setFileContent('');
            setRelationships(null);
        }
    }, [selectedNode, workspaceRoot]);

    useEffect(() => {
        setCurrentScope('');
    }, [selectedIndex]);

    useEffect(() => {
        if (selectedIndex && cyRef.current) {
            setLoading(true);
            setSelectedNode(null);
            setSelectedEdge(null);
            GetGraphData(selectedIndex, currentScope).then((data: any) => {
                if (data && cyRef.current) {
                    setGraphData(data);
                    setLanguage(data.language || 'Go');
                    const cy = cyRef.current;
                    cy.elements().remove();

                    const elements = data.elements.map((el: any) => {
                        if (el.group === 'nodes' && el.data.kind !== 'Folder') {
                            return {
                                ...el,
                                data: {
                                    ...el.data,
                                    value: Math.round(Math.sqrt(el.data.value) * 12 * 100) / 100
                                }
                            };
                        }
                        if (el.group === 'nodes' && el.data.kind === 'Folder') {
                            return {
                                ...el,
                                data: {
                                    ...el.data,
                                    value: 60 // Minimum size for folder containers
                                }
                            };
                        }
                        return el;
                    });

                    cy.add(elements);

                    const layout = cy.layout({
                        name: 'fcose',
                        randomize: true,
                        packComponents: true,
                        nodeRepulsion: () => 8500, // Even more repulsion since clusters are invisible
                        idealEdgeLength: () => 200,
                        sampleSize: 100,
                        nodeSeparation: 250,
                        gravity: 0.25,
                        animate: false,
                        padding: 100,
                        tile: true,
                        tilingPaddingVertical: 120,
                        tilingPaddingHorizontal: 120
                    } as any);

                    layout.run();
                    cy.fit(undefined, 50);

                    if (pendingFocus) {
                        const target = cy.getElementById(pendingFocus);
                        if (!target.empty()) {
                            target.select();
                            setSelectedNode(target.data());
                            cy.animate({
                                center: { eles: target },
                                zoom: 1.5,
                                duration: 500
                            });
                        }
                        setPendingFocus(null);
                    }
                }
                setLoading(false);
            });
        }
    }, [selectedIndex, currentScope]);

    const handleZoomIn = () => {
        cyRef.current?.zoom(cyRef.current.zoom() * 1.5);
    };

    const handleZoomOut = () => {
        cyRef.current?.zoom(cyRef.current.zoom() / 1.5);
    };

    const handleReset = () => {
        cyRef.current?.fit(undefined, 50);
        setSelectedNode(null);
        setSelectedEdge(null);
    };

    const focusSymbol = (query: string) => {
        if (!cyRef.current) return;
        const cy = cyRef.current;

        let target = cy.nodes().filter((n: any) =>
            n.data('name') === query ||
            (n.data('id') && n.data('id') === "sym:" + query) ||
            (n.data('id') && n.data('id') === query)
        );

        if (target.empty()) {
            target = cy.nodes().filter((n: any) =>
                n.data('name').toLowerCase().includes(query.toLowerCase())
            );
        }

        if (!target.empty()) {
            const first = target.first();
            cy.animate({
                center: { eles: first },
                zoom: 1.5,
                duration: 500
            });
            setSelectedNode(first.data());
            first.select();
        }
    };

    const handleCopyForAgent = () => {
        if (!selectedNode) return;
        let snippet = fileContent;
        if (fileContent && selectedNode.startLine && selectedNode.endLine) {
            const lines = fileContent.split('\n');
            snippet = lines.slice(selectedNode.startLine - 1, selectedNode.endLine).join('\n');
        }

        const text = `### Symbol Context: ${selectedNode.name}
- **Path:** ${selectedNode.path}
- **Signature:** \`${selectedNode.name}\`
- **Structure:** ${selectedNode.kind}
- **Snippet:**
\`\`\`${language.toLowerCase()}
${snippet || '// No content available'}
\`\`\`
`;
        navigator.clipboard.writeText(text);
    };

    const handleSearchChange = (val: string) => {
        setSearchTerm(val);
        if (!val.trim()) {
            setSearchResults([]);
            setShowSearchResults(false);
            return;
        }

        SearchSymbols(selectedIndex, val).then(results => {
            setSearchResults(results || []);
            setShowSearchResults((results || []).length > 0);
        }).catch(err => {
            console.error("Global search failed:", err);
        });
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchResults.length > 0) {
            const top = searchResults[0];
            if (top.path !== currentScope) {
                setPendingFocus(top.id);
                setCurrentScope(top.path || '');
            } else {
                focusSymbol(top.id);
            }
            setShowSearchResults(false);
        } else if (searchTerm) {
            focusSymbol(searchTerm);
        }
    };

    return (
        <div className="flex flex-col flex-1 h-full w-full gap-4 animate-in fade-in duration-300 bg-base-100">
            <div className="flex justify-between items-center px-6 pt-5 shrink-0">
                <div className="flex items-center gap-4">
                    <h1 className="text-3xl font-bold font-sans tracking-tight">Code Atlas</h1>
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

                {indexingStatus && (
                    <div className={`p-4 rounded-xl border flex items-center gap-3 animate-in fade-in slide-in-from-top-2 duration-300 shadow-lg ${
                        indexingStatus.type === 'processing' ? 'bg-primary/10 text-primary border-primary/20' :
                        indexingStatus.type === 'success' ? 'bg-success/10 text-success border-success/20' : 
                        'bg-error/10 text-error border-error/20'
                    }`}>
                        <Icon 
                            icon={indexingStatus.type === 'processing' ? 'lucide:loader-2' : indexingStatus.type === 'success' ? 'lucide:check-circle' : 'lucide:alert-circle'} 
                            className={`w-4 h-4 ${indexingStatus.type === 'processing' ? 'animate-spin' : ''}`} 
                        />
                        <span className="text-[11px] font-black uppercase tracking-widest">{indexingStatus.message}</span>
                        {indexingStatus.type !== 'processing' && (
                            <button onClick={() => setIndexingStatus(null)} className="ml-2 hover:opacity-50">
                                <Icon icon="lucide:x" className="w-3.5 h-3.5" />
                            </button>
                        )}
                    </div>
                )}
            </div>

            <div className="flex flex-1 gap-6 px-6 pb-6 overflow-hidden min-h-0">
                <div className="w-80 flex flex-col gap-8 shrink-0 border-r border-base-200 pr-4 overflow-y-auto scrollbar-hide py-4">
                    <div className="flex flex-col gap-4 relative" ref={searchRef}>
                        <div className="text-[10px] uppercase tracking-[0.2em] opacity-40 font-black px-1">Symbol Search</div>
                        <form onSubmit={handleSearch} className="relative">
                            <input
                                type="text"
                                placeholder="Search definitions..."
                                className="input input-bordered w-full bg-base-200/40 focus:bg-base-200 transition-all shadow-sm pl-10 h-10 border-base-300 rounded-lg text-sm"
                                value={searchTerm}
                                onChange={(e) => handleSearchChange(e.target.value)}
                                onFocus={() => searchTerm && setShowSearchResults(true)}
                            />
                            <Icon icon="lucide:search" className="absolute left-3.5 top-1/2 -translate-y-1/2 opacity-30 w-4 h-4" />
                        </form>

                        {showSearchResults && searchResults.length > 0 && (
                            <div className="absolute top-[calc(100%+8px)] left-0 right-0 bg-base-100 border border-base-300 rounded-xl shadow-2xl z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                                <div className="max-h-80 overflow-y-auto scrollbar-hide py-1">
                                    {searchResults.map((res: any) => (
                                        <button
                                            key={res.id}
                                            onClick={() => {
                                                if (res.path !== currentScope) {
                                                    setPendingFocus(res.id);
                                                    setCurrentScope(res.path || '');
                                                } else {
                                                    focusSymbol(res.id);
                                                }
                                                setShowSearchResults(false);
                                            }}
                                            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-primary hover:text-primary-content transition-colors group text-left"
                                        >
                                            <div className="flex items-center gap-3 min-w-0">
                                                <Icon
                                                    icon={res.kind === 'Struct' ? 'lucide:box' : res.kind === 'Function' ? 'lucide:zap' : 'lucide:hash'}
                                                    className="opacity-40 group-hover:opacity-100 w-3.5 h-3.5 shrink-0"
                                                />
                                                <span className="text-xs font-bold truncate">{res.name}</span>
                                            </div>
                                            <span className="text-[10px] uppercase font-black opacity-30 group-hover:opacity-100 bg-base-200/50 group-hover:bg-primary-focus px-1.5 py-0.5 rounded ml-2 shrink-0">
                                                {getKindLabel(res.kind)}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="flex flex-col gap-4">
                        <div className="flex items-center justify-between px-1">
                            <div className="text-[10px] uppercase tracking-[0.2em] opacity-40 font-black">Symbolic Indexes</div>
                            <span className="badge badge-ghost badge-sm opacity-20 font-mono scale-75 border-none">{indexFiles.length}</span>
                        </div>

                        <div className="flex flex-col">
                            {indexFiles.length === 0 ? (
                                <div className="p-8 border border-dashed border-base-300 rounded-xl text-center opacity-20 text-xs">
                                    No indexes found
                                </div>
                            ) : (
                                <div className="flex flex-col">
                                    {renderTree(buildTree(indexFiles, workspaceRoot))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex-1 relative bg-base-100/50 rounded-xl border border-base-200 overflow-hidden shadow-sm flex flex-col min-w-0">
                    {/* Breadcrumbs / Scope Navigation */}
                    {selectedIndex && (
                        <div className="absolute top-6 left-6 right-6 z-30 pointer-events-none">
                            <div className="flex items-center gap-2 bg-base-100/90 backdrop-blur-xl border border-base-300/50 rounded-xl px-4 py-2 shadow-xl ring-1 ring-black/5 pointer-events-auto w-fit max-w-full">
                                <button 
                                    onClick={() => setCurrentScope('')}
                                    className={`btn btn-ghost btn-xs gap-1.5 px-2 hover:bg-primary/10 hover:text-primary transition-colors text-[10px] font-black uppercase tracking-wider ${!currentScope ? 'text-primary' : 'text-base-content/40'}`}
                                >
                                    <Icon icon="lucide:home" className="w-3 h-3" />
                                    Root
                                </button>
                                
                                {currentScope && (
                                    <>
                                        <Icon icon="lucide:chevron-right" className="w-3 h-3 opacity-20" />
                                        <div className="flex items-center gap-1 overflow-hidden">
                                            {currentScope.split('/').map((part, i, arr) => (
                                                <div key={i} className="flex items-center gap-1 shrink-0">
                                                    <button 
                                                        onClick={() => setCurrentScope(arr.slice(0, i + 1).join('/'))}
                                                        className={`btn btn-ghost btn-xs px-2 hover:bg-primary/10 hover:text-primary transition-colors text-[10px] font-black uppercase tracking-wider ${i === arr.length - 1 ? 'text-primary opacity-100' : 'text-base-content/60 opacity-60'}`}
                                                    >
                                                        {part}
                                                    </button>
                                                    {i < arr.length - 1 && <Icon icon="lucide:chevron-right" className="w-3 h-3 opacity-20" />}
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}

                                {graphData?.elements?.filter((e: any) => e.group === 'nodes' && e.data.kind === 'Folder').length > 0 && (
                                    <div className="dropdown ml-4 pointer-events-auto">
                                        <label tabIndex={0} className="btn btn-ghost btn-xs gap-1 opacity-40 hover:opacity-100 transition-all hover:bg-primary/10 hover:text-primary rounded-lg px-2">
                                            <Icon icon="lucide:folder-tree" className="w-3 h-3" />
                                            <span className="text-[10px] font-black uppercase tracking-wider">Jump to</span>
                                            <Icon icon="lucide:chevron-down" className="w-2.5 h-2.5" />
                                        </label>
                                        <ul tabIndex={0} className="dropdown-content z-[100] menu p-2 shadow-2xl bg-base-200/98 backdrop-blur-2xl border border-base-300 rounded-2xl w-64 mt-2 ring-1 ring-black/5 animate-in fade-in slide-in-from-top-2 duration-200">
                                            <li className="menu-title px-3 py-2 text-[9px] uppercase tracking-[0.2em] opacity-40 font-black mb-1">
                                                Subdirectories
                                            </li>
                                            <div className="max-h-64 overflow-y-auto custom-scrollbar pr-1">
                                                {graphData.elements
                                                    .filter((e: any) => e.group === 'nodes' && e.data.kind === 'Folder')
                                                    .sort((a: any, b: any) => a.data.name.localeCompare(b.data.name))
                                                    .map((folder: any) => (
                                                        <li key={folder.data.id} className="mb-0.5 last:mb-0">
                                                            <button 
                                                                onClick={() => {
                                                                    const newScope = folder.data.id.replace('dir:', '');
                                                                    setCurrentScope(newScope);
                                                                    // Close dropdown by blurring active element
                                                                    if (document.activeElement instanceof HTMLElement) {
                                                                        document.activeElement.blur();
                                                                    }
                                                                }}
                                                                className="flex items-center gap-3 py-2.5 px-3 hover:bg-primary/10 hover:text-primary rounded-xl transition-all group"
                                                            >
                                                                <div className="w-8 h-8 rounded-lg bg-base-300/50 flex items-center justify-center shrink-0 group-hover:bg-primary/20 transition-colors">
                                                                    <Icon icon="lucide:folder" className="w-4 h-4 opacity-70 group-hover:opacity-100" />
                                                                </div>
                                                                <div className="flex flex-col items-start min-w-0">
                                                                    <span className="text-xs font-bold truncate w-full">{folder.data.name}</span>
                                                                    <span className="text-[9px] opacity-40 font-mono truncate w-full">{folder.data.id.replace('dir:', '')}</span>
                                                                </div>
                                                                <Icon icon="lucide:arrow-right" className="ml-auto w-3.5 h-3.5 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                                                            </button>
                                                        </li>
                                                    ))}
                                            </div>
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {loading && (
                        <div className="absolute inset-0 z-50 flex items-center justify-center bg-base-100/50 backdrop-blur-sm">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                        </div>
                    )}
                    <div
                        ref={chartRef}
                        className="flex-1 w-full h-full cursor-grab active:cursor-grabbing"
                        style={{ pointerEvents: 'auto' }}
                    />

                    <div className="absolute left-6 bottom-6 flex flex-col gap-4 pointer-events-none z-10">
                        <div className="flex items-center gap-1 bg-base-100/90 backdrop-blur-xl border border-base-300/50 rounded-xl p-1.5 shadow-xl ring-1 ring-black/5 pointer-events-auto w-fit">
                            <button onClick={handleZoomIn} className="btn btn-ghost btn-sm btn-square hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60" title="Zoom In">
                                <Icon icon="lucide:zoom-in" className="w-4 h-4" />
                            </button>
                            <button onClick={handleZoomOut} className="btn btn-ghost btn-sm btn-square hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60" title="Zoom Out">
                                <Icon icon="lucide:zoom-out" className="w-4 h-4" />
                            </button>
                            <div className="w-px h-4 bg-base-content/10 mx-0.5" />
                            <button onClick={handleReset} className="btn btn-ghost btn-sm px-3 gap-2 hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60 text-[10px] font-black uppercase tracking-wider" title="Reset View">
                                <Icon icon="lucide:refresh-cw" className="w-3.5 h-3.5" />
                                Reset
                            </button>
                        </div>

                        <div className="bg-base-100/90 backdrop-blur-xl border border-base-300/50 rounded-xl p-4 shadow-xl ring-1 ring-black/5 pointer-events-auto">
                            <div className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Symbol Legend</div>
                            <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#f59e0b] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">{getKindLabel('Struct', true)}</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#10b981] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">{getKindLabel('Interface', true)}</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#3b82f6] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">{getKindLabel('Function', true)}</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#6b7280] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">{getKindLabel('Variable', true)}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {!selectedIndex && indexFiles.length > 0 && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="text-center opacity-10">
                                <Icon icon="lucide:network" className="text-8xl mx-auto mb-4" />
                                <p className="text-xl font-black">Select an index to explore</p>
                            </div>
                        </div>
                    )}

                    {(selectedNode || selectedEdge) && (
                        <div className="absolute right-6 top-6 bottom-6 w-[380px] bg-base-200 border border-base-300 shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-right duration-300 z-40 rounded-xl p-0.5">
                            <div className="flex flex-col h-full bg-base-100 rounded-[10px] overflow-hidden">
                                <div className="flex items-center justify-between px-6 py-4 border-b border-base-300/50 bg-base-200/50">
                                    <div className="flex items-center gap-2">
                                        <Icon icon="lucide:info" className="w-3.5 h-3.5 text-primary" />
                                        <span className="text-[10px] font-black uppercase tracking-widest text-primary">Details</span>
                                    </div>
                                    <button onClick={() => { setSelectedNode(null); setSelectedEdge(null); }} className="btn btn-ghost btn-xs btn-square hover:bg-error/10 hover:text-error">
                                        <Icon icon="lucide:x" className="w-4 h-4" />
                                    </button>
                                </div>

                                <div className="flex-1 flex flex-col overflow-hidden">
                                    <div className="flex-1 overflow-y-auto">
                                        {selectedNode ? (
                                            <>
                                                <div className="p-6 border-b border-base-300/50 flex justify-between items-start bg-base-200/50">
                                                    <div className="min-w-0 pr-4">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <div className="badge badge-primary badge-outline text-[10px] font-black px-2 h-5 opacity-60 uppercase tracking-widest rounded-md">
                                                                {getKindLabel(selectedNode.kind)}
                                                            </div>
                                                        </div>
                                                        <h2 className="text-xl font-black leading-tight tracking-tighter break-words" title={selectedNode.name}>
                                                            {selectedNode.name}
                                                        </h2>
                                                        <div className="flex items-center gap-2 mt-4">
                                                            <div className="flex flex-col bg-base-200/80 px-3 py-1.5 rounded-lg border border-base-300/50 min-w-[70px]">
                                                                <span className="text-[8px] uppercase opacity-40 font-black tracking-widest leading-none mb-1">Impact</span>
                                                                <span className="text-sm font-mono font-bold leading-none text-primary">{selectedNode.value}</span>
                                                            </div>
                                                            {selectedNode.loc > 0 && (
                                                                <div className="flex flex-col bg-base-200/80 px-3 py-1.5 rounded-lg border border-base-300/50 min-w-[70px]">
                                                                    <span className="text-[8px] uppercase opacity-40 font-black tracking-widest leading-none mb-1">Lines</span>
                                                                    <span className="text-sm font-mono font-bold leading-none">{selectedNode.loc}</span>
                                                                </div>
                                                            )}
                                                        </div>
                                                        <div className="flex items-center gap-2 mt-4 opacity-30">
                                                            <Icon icon="lucide:file-text" className="w-3.5 h-3.5" />
                                                            <span className="text-[10px] font-mono truncate max-w-[200px]">{selectedNode.path || 'Virtual Node'}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="p-6 space-y-8 flex-1 overflow-y-auto">
                                                    {selectedNode.path && fileContent && (
                                                        <section>
                                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Signature</h3>
                                                            <div className="p-0 overflow-hidden bg-[#1e1e1e] rounded-xl border border-base-300/50 w-full max-w-full">
                                                                <SyntaxHighlighter
                                                                    language={language.toLowerCase()}
                                                                    style={vscDarkPlus}
                                                                    wrapLongLines={true}
                                                                    customStyle={{
                                                                        margin: 0,
                                                                        padding: '1.25rem',
                                                                        backgroundColor: 'transparent',
                                                                        fontSize: '11px',
                                                                        lineHeight: '1.6',
                                                                    }}
                                                                >
                                                                    {fileContent.split('\n').slice(selectedNode.startLine - 1, (selectedNode.endLine || selectedNode.startLine) + 1).join('\n')}
                                                                </SyntaxHighlighter>
                                                            </div>
                                                        </section>
                                                    )}

                                                    {selectedNode.docstring && (
                                                        <section>
                                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Documentation</h3>
                                                            <div className="p-0 overflow-hidden bg-base-200/50 rounded-xl border border-base-300/50 w-full max-w-full">
                                                                <ReactMarkdown
                                                                    components={{
                                                                        code({ node, inline, className, children, ...props }: any) {
                                                                            const match = /language-(\w+)/.exec(className || '');
                                                                            return !inline && match ? (
                                                                                <SyntaxHighlighter
                                                                                    {...props}
                                                                                    children={String(children).replace(/\n$/, '')}
                                                                                    style={vscDarkPlus}
                                                                                    language={match[1]}
                                                                                    wrapLongLines={true}
                                                                                    customStyle={{
                                                                                        margin: 0,
                                                                                        padding: '1.25rem',
                                                                                        backgroundColor: 'transparent',
                                                                                        fontSize: '11px',
                                                                                        lineHeight: '1.6',
                                                                                        whiteSpace: 'pre-wrap',
                                                                                        wordBreak: 'break-word'
                                                                                    }}
                                                                                    codeTagProps={{
                                                                                        style: {
                                                                                            whiteSpace: 'pre-wrap',
                                                                                            wordBreak: 'break-word',
                                                                                            display: 'block'
                                                                                        }
                                                                                    }}
                                                                                />
                                                                            ) : (
                                                                                <code className={`${className} bg-base-300 px-1 py-0.5 rounded text-[11px] font-mono`} {...props}>
                                                                                    {children}
                                                                                </code>
                                                                            )
                                                                        }
                                                                    }}
                                                                >
                                                                    {selectedNode.docstring}
                                                                </ReactMarkdown>
                                                            </div>
                                                        </section>
                                                    )}

                                                    {selectedNode.members && selectedNode.members.length > 0 && (
                                                        <section>
                                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Structure</h3>
                                                            <div className="grid grid-cols-1 gap-2">
                                                                {selectedNode.members.map((m: any, i: number) => (
                                                                    <div key={i} className="flex items-center justify-between p-3.5 bg-base-200/30 rounded-xl border border-base-300/50 group hover:border-primary/30 transition-all shadow-sm hover:shadow-md">
                                                                        <div className="flex items-center gap-3">
                                                                            <Icon icon={m.kind === 'Field' ? 'lucide:box' : 'lucide:function-square'} className={`w-3.5 h-3.5 ${m.kind === 'Field' ? 'text-blue-400' : 'text-purple-400'} opacity-60`} />
                                                                            <span className="text-[11px] font-bold tracking-tight">{m.name}</span>
                                                                        </div>
                                                                        <span className="text-[10px] font-mono opacity-30 group-hover:opacity-60 transition-opacity uppercase font-black">{m.type || 'any'}</span>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </section>
                                                    )}

                                                    {loadingRelationships && (
                                                        <section className="animate-pulse">
                                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Analyzing Connections</h3>
                                                            <div className="space-y-3">
                                                                <div className="h-4 bg-base-300/50 rounded w-3/4"></div>
                                                                <div className="h-4 bg-base-300/50 rounded w-1/2"></div>
                                                            </div>
                                                        </section>
                                                    )}

                                                    {relationships && (relationships.callers.length > 0 || relationships.callees.length > 0) && (
                                                        <>
                                                            {relationships.callers.length > 0 && (
                                                                <section>
                                                                    <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Callers ({relationships.callers.length})</h3>
                                                                    <div className="grid grid-cols-1 gap-2">
                                                                        {relationships.callers.map((r: any, i: number) => (
                                                                            <button 
                                                                                key={i} 
                                                                                onClick={() => {
                                                                                    if (r.path !== currentScope) {
                                                                                        setPendingFocus(r.id);
                                                                                        setCurrentScope(r.path || '');
                                                                                    } else {
                                                                                        focusSymbol(r.id);
                                                                                    }
                                                                                }}
                                                                                className="flex flex-col items-start p-3.5 bg-base-200/30 rounded-xl border border-base-300/50 group hover:border-primary/30 transition-all shadow-sm hover:shadow-md text-left"
                                                                            >
                                                                                <div className="flex items-center gap-3 mb-1">
                                                                                    <Icon icon="lucide:arrow-up-left" className="w-3 h-3 text-primary opacity-60" />
                                                                                    <span className="text-[11px] font-bold tracking-tight">{r.name}</span>
                                                                                </div>
                                                                                <span className="text-[9px] font-mono opacity-30 truncate w-full">{r.path || 'root'}</span>
                                                                            </button>
                                                                        ))}
                                                                    </div>
                                                                </section>
                                                            )}

                                                            {relationships.callees.length > 0 && (
                                                                <section>
                                                                    <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-4 px-1">Callees ({relationships.callees.length})</h3>
                                                                    <div className="grid grid-cols-1 gap-2">
                                                                        {relationships.callees.map((r: any, i: number) => (
                                                                            <button 
                                                                                key={i} 
                                                                                onClick={() => {
                                                                                    if (r.path !== currentScope) {
                                                                                        setPendingFocus(r.id);
                                                                                        setCurrentScope(r.path || '');
                                                                                    } else {
                                                                                        focusSymbol(r.id);
                                                                                    }
                                                                                }}
                                                                                className="flex flex-col items-start p-3.5 bg-base-200/30 rounded-xl border border-base-300/50 group hover:border-primary/30 transition-all shadow-sm hover:shadow-md text-left"
                                                                            >
                                                                                <div className="flex items-center gap-3 mb-1">
                                                                                    <Icon icon="lucide:arrow-down-right" className="w-3 h-3 text-success opacity-60" />
                                                                                    <span className="text-[11px] font-bold tracking-tight">{r.name}</span>
                                                                                </div>
                                                                                <span className="text-[9px] font-mono opacity-30 truncate w-full">{r.path || 'root'}</span>
                                                                            </button>
                                                                        ))}
                                                                    </div>
                                                                </section>
                                                            )}
                                                        </>
                                                    )}
                                                </div>

                                                <div className="p-6 bg-base-200/50 border-t border-base-300/50 flex gap-3">
                                                    <button
                                                        onClick={handleCopyForAgent}
                                                        className="flex-1 btn btn-outline btn-primary h-12 gap-3 rounded-xl text-[10px] font-black uppercase tracking-widest"
                                                    >
                                                        <Icon icon="lucide:clipboard-copy" className="w-4 h-4" />
                                                        Copy for Agent
                                                    </button>
                                                    <button
                                                        onClick={() => setIsSourceExpanded(true)}
                                                        className="btn btn-primary btn-square w-12 h-12 rounded-xl"
                                                        title="View Source"
                                                    >
                                                        <Icon icon="lucide:code-2" className="w-5 h-5" />
                                                    </button>
                                                    <button
                                                        onClick={() => focusSymbol(selectedNode.id)}
                                                        className="btn btn-primary btn-square w-12 h-12 rounded-xl"
                                                        title="Pinpoint Logic"
                                                    >
                                                        <Icon icon="lucide:locate-fixed" className="w-5 h-5" />
                                                    </button>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <div className="p-6 border-b border-base-300/50 bg-base-200/50">
                                                    <div className="flex items-center gap-3 mb-2">
                                                        <Icon icon="lucide:git-pull-request" className="w-4 h-4 text-primary" />
                                                        <h2 className="text-xl font-black tracking-tighter">Connection</h2>
                                                    </div>
                                                    <p className="text-[10px] font-bold text-base-content/40 uppercase tracking-widest">Symbol Flow Analysis</p>
                                                </div>

                                                <div className="p-6 space-y-8 flex-1 overflow-y-auto scrollbar-hide">
                                                    <section className="bg-base-200/30 rounded-xl border border-base-300/50 p-5 relative">
                                                        <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-3 ml-1">Origin</h3>
                                                        <div className="font-mono text-xs font-bold text-primary break-all leading-relaxed bg-base-100 p-3 rounded-lg border border-base-300/30 shadow-sm">
                                                            {selectedEdge.source.replace(/^sym:\s*/, '').replace(/^scip-[^\s]+\s+[^\s]+\s+/, '')}
                                                        </div>
                                                    </section>

                                                    <div className="flex justify-center -my-6 relative z-10">
                                                        <div className="bg-primary p-2.5 rounded-full border-4 border-base-100 shadow-xl">
                                                            <Icon icon="lucide:chevron-down" className="text-primary-content w-4 h-4" />
                                                        </div>
                                                    </div>

                                                    <section className="bg-base-200/30 rounded-xl border border-base-300/50 p-5 relative">
                                                        <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-3 ml-1">Destination</h3>
                                                        <div className="font-mono text-xs font-bold text-success break-all leading-relaxed bg-base-100 p-3 rounded-lg border border-base-300/30 shadow-sm">
                                                            {selectedEdge.target.replace(/^sym:\s*/, '').replace(/^scip-[^\s]+\s+[^\s]+\s+/, '')}
                                                        </div>
                                                    </section>

                                                    <section>
                                                        <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-3 px-1">Flow Type</h3>
                                                        <div className="p-4 bg-primary/10 rounded-xl border border-primary/20 text-xs font-black italic text-primary/80 break-words flex items-center gap-3">
                                                            <Icon icon="lucide:git-branch" className="w-4 h-4" />
                                                            {selectedEdge.label || 'Static Dependency'}
                                                        </div>
                                                    </section>
                                                </div>
                                                <div className="p-6 bg-base-200/50 border-t border-base-300/50 flex flex-col gap-3">
                                                    <button
                                                        onClick={() => focusSymbol(selectedEdge.source.replace('sym:', ''))}
                                                        className="btn btn-outline h-12 rounded-xl gap-3 border-base-300 hover:bg-base-200 transition-all font-bold text-sm"
                                                    >
                                                        <Icon icon="lucide:arrow-up-left" className="w-4 h-4" />
                                                        Jump to Origin
                                                    </button>
                                                    <button
                                                        onClick={() => focusSymbol(selectedEdge.target.replace('sym:', ''))}
                                                        className="btn btn-outline h-12 rounded-xl gap-3 border-base-300 hover:bg-base-200 transition-all font-bold text-sm"
                                                    >
                                                        <Icon icon="lucide:arrow-down-right" className="w-4 h-4" />
                                                        Jump to Destination
                                                    </button>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {isSourceExpanded && (
                        <div className="absolute inset-0 z-50 bg-base-100 flex flex-col animate-in fade-in duration-200">
                            <div className="flex items-center justify-between p-4 border-b border-base-300 bg-base-200/50">
                                <div className="flex items-center gap-4">
                                    <div className="flex flex-col">
                                        <h2 className="text-lg font-black tracking-tighter leading-none">{selectedNode?.name}</h2>
                                        <span className="text-[10px] font-mono opacity-40 mt-1">{selectedNode?.path}</span>
                                    </div>
                                    <div className="badge badge-primary badge-outline text-[10px] font-black px-2 h-5 opacity-60 uppercase tracking-widest rounded-md">
                                        {getKindLabel(selectedNode?.kind)}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={handleCopyForAgent}
                                        className="btn btn-sm btn-outline btn-primary gap-2 text-[10px] font-black uppercase tracking-widest"
                                    >
                                        <Icon icon="lucide:clipboard-copy" className="w-3.5 h-3.5" />
                                        Copy for Agent
                                    </button>
                                    <button
                                        onClick={() => setIsSourceExpanded(false)}
                                        className="btn btn-sm btn-ghost btn-square"
                                    >
                                        <Icon icon="lucide:x" className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>
                            <div className="flex-1 overflow-hidden bg-[#1e1e1e]">
                                <Editor
                                    height="100%"
                                    defaultLanguage={language.toLowerCase()}
                                    theme="vs-dark"
                                    value={fileContent}
                                    options={{
                                        readOnly: true,
                                        minimap: { enabled: true },
                                        scrollBeyondLastLine: false,
                                        fontSize: 13,
                                        lineNumbers: 'on',
                                        wordWrap: 'on',
                                        renderWhitespace: 'none',
                                    }}
                                    onMount={(editor) => {
                                        editorRef.current = editor; // Keep reference to the expanded editor
                                        if (selectedNode?.startLine) {
                                            editor.revealLineInCenter(selectedNode.startLine);
                                            (editor as any)._decorations = editor.deltaDecorations([], [
                                                {
                                                    range: { startLineNumber: selectedNode.startLine, startColumn: 1, endLineNumber: selectedNode.endLine || selectedNode.startLine, endColumn: 100 },
                                                    options: {
                                                        isWholeLine: true,
                                                        className: 'bg-primary/5',
                                                        linesDecorationsClassName: 'symbol-gutter-highlight',
                                                    }
                                                }
                                            ]);
                                        }
                                    }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
