import { useEffect, useState, useRef } from 'react';
import { Icon } from '@iconify/react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { SearchForIndexes, GetGraphData, GetWorkspaces } from '../wailsjs/go/main/App';
import { EventsOn, EventsOff } from '../wailsjs/runtime/runtime';
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

    const renderTree = (nodes: any[], depth = 0) => {
        return nodes.sort((a,b) => (a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === 'folder' ? -1 : 1)).map(node => (
            <div key={node.path}>
                {node.kind === 'folder' ? (
                    <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-black opacity-30 uppercase tracking-[0.2em] select-none mt-2 first:mt-0">
                        <div style={{ width: depth * 12 }} />
                        <Icon icon="lucide:folder" className="w-2.5 h-2.5" />
                        {node.name}
                    </div>
                ) : (
                    <button
                        key={node.path}
                        onClick={() => setSelectedIndex(node.path)}
                        className={`flex items-center gap-2 w-full px-3 py-1.5 text-[11px] rounded-lg transition-all text-left group ${
                            selectedIndex === node.path 
                            ? 'bg-primary/10 text-primary font-bold border border-primary/20' 
                            : 'hover:bg-base-200 border border-transparent opacity-60 hover:opacity-100'
                        }`}
                        style={{ paddingLeft: depth * 12 + 12 }}
                    >
                        <Icon icon="lucide:file-code" className={`w-3 h-3 ${selectedIndex === node.path ? 'opacity-100' : 'opacity-30 group-hover:opacity-100'}`} />
                        <span className="truncate">{node.name}</span>
                        {selectedIndex === node.path && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary" />}
                    </button>
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

        return () => EventsOff('workspace-updated');
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
                            'background-opacity': 0,
                            'border-width': 0,
                            'label': '',
                            'padding': '40px', // Extra padding for logical clustering
                            'z-index': 1,
                            'events': 'no' // Folders are transparent to clicks/hovers
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
        if (selectedIndex && cyRef.current) {
            setLoading(true);
            setSelectedNode(null);
            setSelectedEdge(null);
            GetGraphData(selectedIndex).then((data) => {
                if (data && cyRef.current) {
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
                                    value: 0 // Folders take size from children
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
                }
                setLoading(false);
            });
        }
    }, [selectedIndex]);

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

    const handleSearchChange = (val: string) => {
        setSearchTerm(val);
        if (!val.trim() || !cyRef.current) {
            setSearchResults([]);
            setShowSearchResults(false);
            return;
        }

        const cy = cyRef.current;
        const allNodes = cy.nodes().filter((n: any) => n.data('kind') !== 'Folder');
        const query = val.toLowerCase();

        const filtered = allNodes.map((n: any) => n.data())
            .filter((data: any) => data.name.toLowerCase().includes(query))
            .sort((a: any, b: any) => {
                const aName = a.name.toLowerCase();
                const bName = b.name.toLowerCase();
                
                // Exact match first
                if (aName === query && bName !== query) return -1;
                if (bName === query && aName !== query) return 1;
                
                // Prefix match next
                const aPrefix = aName.startsWith(query);
                const bPrefix = bName.startsWith(query);
                if (aPrefix && !bPrefix) return -1;
                if (bPrefix && !aPrefix) return 1;
                
                return aName.localeCompare(bName);
            })
            .slice(0, 15);

        setSearchResults(filtered);
        setShowSearchResults(filtered.length > 0);
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchResults.length > 0) {
            const top = searchResults[0];
            focusSymbol(top.id);
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
                                                focusSymbol(res.id);
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
                                            <span className="text-[9px] uppercase font-black opacity-30 group-hover:opacity-100 bg-base-200/50 group-hover:bg-primary-focus px-1.5 py-0.5 rounded ml-2 shrink-0">
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
                            <div className="text-[9px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Symbol Legend</div>
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
                                            <button onClick={() => setSelectedNode(null)} className="btn btn-ghost btn-xs btn-circle shrink-0 hover:bg-error/10 hover:text-error transition-colors">
                                                <Icon icon="lucide:x" className="w-4 h-4" />
                                            </button>
                                        </div>

                                        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
                                            {selectedNode.docstring && (
                                                <section>
                                                    <h3 className="text-[10px] font-black uppercase tracking-widest opacity-30 mb-3 ml-1">Documentation</h3>
                                                    <div className="prose prose-sm prose-invert max-w-none p-4 rounded-xl border border-base-300/50 bg-base-200/30 text-xs italic opacity-90">
                                                        <ReactMarkdown
                                                            components={{
                                                                code({ node, inline, className, children, ...props }: any) {
                                                                    const match = /language-(\w+)/.exec(className || '');
                                                                    return !inline && match ? (
                                                                        <SyntaxHighlighter
                                                                            style={vscDarkPlus}
                                                                            language={match[1]}
                                                                            PreTag="div"
                                                                            wrapLongLines={true}
                                                                            customStyle={{
                                                                                margin: 0,
                                                                                padding: '12px',
                                                                                background: 'rgba(0,0,0,0.2)',
                                                                                fontSize: '11px',
                                                                                lineHeight: '1.5',
                                                                                overflowX: 'hidden',
                                                                                whiteSpace: 'pre-wrap',
                                                                                wordBreak: 'break-all'
                                                                            }}
                                                                            codeTagProps={{
                                                                                style: {
                                                                                    whiteSpace: 'pre-wrap',
                                                                                    wordBreak: 'break-all',
                                                                                    display: 'block'
                                                                                }
                                                                            }}
                                                                        >
                                                                            {String(children).replace(/\n$/, '')}
                                                                        </SyntaxHighlighter>
                                                                    ) : (
                                                                        <code className={className} {...props}>
                                                                            {children}
                                                                        </code>
                                                                    );
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
                                                    <div className="flex items-center justify-between mb-3 px-1">
                                                        <h3 className="text-[10px] font-black uppercase tracking-widest opacity-30">Members</h3>
                                                        <span className="badge badge-ghost badge-sm opacity-30 font-mono font-bold">{selectedNode.members.length}</span>
                                                    </div>
                                                    <div className="flex flex-col gap-1.5 bg-base-200/20 p-1.5 rounded-xl border border-base-300/50">
                                                        {selectedNode.members.map((m: any, idx: number) => (
                                                            <button
                                                                key={idx}
                                                                onClick={() => focusSymbol(m.symbol ? "sym:" + m.symbol : m.name)}
                                                                className="flex items-center justify-between p-3.5 hover:bg-primary hover:text-primary-content rounded-lg transition-all duration-200 text-xs text-left group border border-transparent shadow-sm"
                                                            >
                                                                <div className="flex flex-col gap-1 min-w-0">
                                                                    <div className="flex items-center gap-2">
                                                                        <Icon icon={m.kind === 'Method' ? "lucide:zap" : "lucide:hash"} className="opacity-20 group-hover:opacity-100 w-3.5 h-3.5" />
                                                                        <span className="font-bold opacity-80 group-hover:opacity-100 truncate">
                                                                            {m.name}{m.kind === 'Method' ? '()' : ''}
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                                <span className="opacity-30 text-[9px] uppercase font-mono bg-base-300/50 px-2 py-0.5 rounded group-hover:bg-primary-focus group-hover:opacity-100 shrink-0 h-fit">
                                                                    {getKindLabel(m.kind)}
                                                                </span>
                                                            </button>
                                                        ))}
                                                    </div>
                                                </section>
                                            )}
                                        </div>

                                        <div className="p-6 bg-base-200/50 border-t border-base-300/50">
                                            <button
                                                onClick={() => focusSymbol(selectedNode.id)}
                                                className="btn btn-primary w-full h-12 gap-3 shadow-lg shadow-primary/20 rounded-xl text-sm font-bold"
                                            >
                                                <Icon icon="lucide:target" className="w-4 h-4" />
                                                Pinpoint Logic
                                            </button>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="p-6 border-b border-base-300/50 flex justify-between items-start bg-base-200/50">
                                            <div>
                                                <div className="badge badge-secondary badge-outline text-[10px] font-black px-2 h-5 opacity-60 uppercase tracking-widest mb-1 rounded-md">Relationship</div>
                                                <h2 className="text-xl font-black tracking-tighter">Call Pipeline</h2>
                                            </div>
                                            <button onClick={() => setSelectedEdge(null)} className="btn btn-ghost btn-xs btn-circle shrink-0 hover:bg-error/10 hover:text-error transition-colors">
                                                <Icon icon="lucide:x" className="w-4 h-4" />
                                            </button>
                                        </div>
                                        <div className="flex-1 p-6 space-y-6 overflow-y-auto scrollbar-hide">
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
                                                    {selectedEdge.label.split('(').pop()?.replace(')', '') || 'Direct Link'}
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
                    )}
                </div>
            </div>
        </div>
    );
}
