import { useEffect, useState, useRef } from 'react';
import { Icon } from '@iconify/react';
import * as echarts from 'echarts';
import { SearchForIndexes, GetGraphData } from '../wailsjs/go/main/App';

type AtlasProps = {
    workspaceRoot: string;
};

export default function Atlas({ workspaceRoot }: AtlasProps) {
    const chartRef = useRef<HTMLDivElement>(null);
    const chartInstance = useRef<echarts.ECharts | null>(null);
    const [indexFiles, setIndexFiles] = useState<string[]>([]);
    const [selectedIndex, setSelectedIndex] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedNode, setSelectedNode] = useState<any>(null);
    const [selectedEdge, setSelectedEdge] = useState<any>(null);

    useEffect(() => {
        if (workspaceRoot) {
            SearchForIndexes(workspaceRoot).then((files) => {
                setIndexFiles(files || []);
                if (files && files.length > 0) {
                    setSelectedIndex(files[0]);
                }
            });
        }
    }, [workspaceRoot]);

    useEffect(() => {
        if (chartRef.current && !chartInstance.current) {
            chartInstance.current = echarts.init(chartRef.current);
            
            chartInstance.current.on('click', (params: any) => {
                if (params.dataType === 'node') {
                    setSelectedEdge(null);
                    setSelectedNode(params.data);
                } else if (params.dataType === 'edge') {
                    setSelectedNode(null);
                    setSelectedEdge(params.data);
                }
            });
            
            // Background click deselects
            chartInstance.current.getZr().on('click', (event: any) => {
                if (!event.target) {
                    setSelectedNode(null);
                    setSelectedEdge(null);
                }
            });
        }

        const resizeObserver = new ResizeObserver(() => {
            chartInstance.current?.resize();
        });

        if (chartRef.current) {
            resizeObserver.observe(chartRef.current);
        }

        return () => {
            resizeObserver.disconnect();
            chartInstance.current?.dispose();
            chartInstance.current = null;
        };
    }, []);

    useEffect(() => {
        if (selectedIndex) {
            setLoading(true);
            setSelectedNode(null);
            setSelectedEdge(null);
            GetGraphData(selectedIndex).then((data) => {
                if (data && chartInstance.current) {
                    // Legend is fixed to Code Kinds (not directories)
                    const legendData = ['Struct', 'Interface', 'Function', 'Variable'];
                    
                    const option = {
                        title: {
                            text: 'Hotpath Structural Atlas (v2)',
                            subtext: selectedIndex.split(/[\\/]/).pop(),
                            top: 'bottom',
                            left: 'right',
                            textStyle: { color: 'rgba(255,255,255,0.3)', fontSize: 11 }
                        },
                        tooltip: {
                            show: true,
                            trigger: 'item',
                            formatter: (params: any) => {
                                if (params.dataType === 'node') {
                                    return `<strong>[${params.data.kind}] ${params.data.name}</strong><br/>Score: ${params.data.value}<br/>${params.data.path || ''}`;
                                }
                                return null; // Disable edge tooltips as requested
                            }
                        },
                        legend: {
                            data: legendData,
                            orient: 'vertical',
                            left: 10,
                            top: 40,
                            textStyle: { fontSize: 10, color: '#aaa' }
                        },
                        series: [
                            {
                                type: 'graph',
                                layout: 'force',
                                data: data.nodes.map((n: any) => ({
                                    ...n,
                                    symbolSize: Math.sqrt(n.value) * 4.5, // Normalized hotpath sizing
                                    itemStyle: {
                                        color: n.kind === 'Struct' ? '#f59e0b' : 
                                               n.kind === 'Interface' ? '#10b981' :
                                               n.kind === 'Function' ? '#3b82f6' : '#6b7280'
                                    },
                                    name: n.name // Keep name for label
                                })),
                                links: data.links,
                                categories: data.categories, // Used for force layout clustering
                                roam: true,
                                edgeSymbol: ['none', 'arrow'],
                                edgeSymbolSize: [0, 8],
                                edgeLabel: { show: false }, // Remove clutter
                                label: {
                                    position: 'right',
                                    formatter: '{b}',
                                    show: true,
                                    fontSize: 9,
                                    color: 'rgba(255,255,255,0.4)',
                                    minMargin: 5
                                },
                                lineStyle: {
                                    color: 'source',
                                    curveness: 0.1,
                                    opacity: 0.1,
                                    width: 1.5
                                },
                                emphasis: {
                                    focus: 'adjacency',
                                    lineStyle: { width: 4, opacity: 0.8 },
                                    label: { show: true, fontWeight: 'bold', fontSize: 11, color: '#fff' }
                                },
                                force: {
                                    repulsion: 3000,
                                    gravity: 0.05,
                                    edgeLength: [60, 180],
                                    layoutAnimation: data.nodes.length < 1500
                                },
                                draggable: true,
                                select: {
                                    itemStyle: { borderColor: '#fff', borderWidth: 2, shadowBlur: 10, shadowColor: '#fff' },
                                    lineStyle: { color: '#fff', width: 3, opacity: 1 }
                                }
                            }
                        ]
                    };
                    
                    chartInstance.current.clear();
                    setTimeout(() => {
                        if (chartInstance.current) {
                            chartInstance.current.setOption(option as any, true);
                        }
                    }, 0);
                }
                setLoading(false);
            });
        }
    }, [selectedIndex]);

    const focusSymbol = (query: string) => {
        if (!chartInstance.current) return;
        const option = chartInstance.current.getOption() as any;
        const nodes = option.series[0].data;
        
        let idx = nodes.findIndex((n: any) => n.name === query || (n.id && n.id === "sym:" + query));
        if (idx === -1) {
            idx = nodes.findIndex((n: any) => n.name.toLowerCase().includes(query.toLowerCase()));
        }

        if (idx !== -1) {
            const node = nodes[idx];
            // Zoom and center
            chartInstance.current.setOption({
                series: [{
                    center: [node.x, node.y],
                    zoom: 2,
                    data: nodes // Keep existing data
                }]
            });

            chartInstance.current.dispatchAction({
                type: 'focusNodeAdjacency',
                seriesIndex: 0,
                dataIndex: idx
            });
            chartInstance.current.dispatchAction({
                type: 'select',
                seriesIndex: 0,
                dataIndex: idx
            });
            setSelectedNode(node);
        }
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (chartInstance.current && searchTerm) {
            const option = chartInstance.current.getOption() as any;
            const nodes = option.series[0].data;
            const targetIdx = nodes.findIndex((n: any) => 
                n.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                (n.path && n.path.toLowerCase().includes(searchTerm.toLowerCase()))
            );

            if (targetIdx !== -1) {
                const node = nodes[targetIdx];
                // Zoom and center
                chartInstance.current.setOption({
                    series: [{
                        center: [node.x, node.y],
                        zoom: 2,
                        data: nodes
                    }]
                });

                chartInstance.current.dispatchAction({
                    type: 'focusNodeAdjacency',
                    seriesIndex: 0,
                    dataIndex: targetIdx
                });
                chartInstance.current.dispatchAction({
                    type: 'select',
                    seriesIndex: 0,
                    dataIndex: targetIdx
                });
                setSelectedNode(node);
            }
        }
    };

    return (
        <div className="flex flex-col flex-1 h-full w-full gap-4 animate-in fade-in duration-300 relative">
            <div className="flex justify-between items-center px-2">
                <div className="flex items-center gap-3">
                    <h1 className="text-3xl font-bold font-sans tracking-tight">Code Atlas</h1>
                    <div className="badge badge-outline badge-sm opacity-50 font-mono">{workspaceRoot}</div>
                </div>

                <form onSubmit={handleSearch} className="relative">
                    <input
                        type="text"
                        placeholder="Search symbols..."
                        className="input input-sm input-bordered w-64 pr-8 focus:input-primary transition-all bg-base-100"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                    <Icon icon="lucide:search" className="absolute right-2 top-1/2 -translate-y-1/2 opacity-50" />
                </form>
            </div>

            <div className="flex flex-1 gap-4 overflow-hidden min-h-0">
                {/* Sidebar: SCIP Files */}
                <div className="w-64 flex flex-col gap-2 overflow-y-auto pr-2 border-r border-base-content/5 flex-shrink-0 lg:flex-shrink">
                    <h3 className="text-xs font-bold uppercase tracking-wider opacity-50 mb-2 px-2">Symbolic Indexes</h3>
                    {indexFiles.length === 0 ? (
                        <div className="p-4 text-sm opacity-50 italic">No .scip files found in this workspace.</div>
                    ) : (
                        indexFiles.map((file) => (
                            <button
                                key={file}
                                onClick={() => setSelectedIndex(file)}
                                className={`text-left p-2 rounded-lg text-xs transition-colors truncate ${
                                    selectedIndex === file 
                                    ? 'bg-primary text-primary-content shadow-md' 
                                    : 'hover:bg-base-300 opacity-70 hover:opacity-100'
                                }`}
                                title={file}
                            >
                                <Icon icon="lucide:file-code" className="inline mr-2" />
                                {file.split(/[\\/]/).pop()}
                            </button>
                        ))
                    )}
                </div>

                {/* Main Canvas */}
                <div className="flex-1 relative bg-base-100 rounded-2xl border border-base-200 overflow-hidden shadow-inner flex flex-col min-w-0">
                    {loading && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center bg-base-100/50 backdrop-blur-sm">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                        </div>
                    )}
                    <div ref={chartRef} className="flex-1 w-full" />
                    
                    {/* Floating Legend Overlay */}
                    <div className="absolute left-6 bottom-6 bg-base-200/80 backdrop-blur-md border border-base-300 rounded-xl p-3 shadow-lg pointer-events-none flex flex-col gap-2 z-10 transition-all hover:bg-base-200">
                        <div className="text-[9px] font-bold uppercase tracking-widest opacity-40 mb-1">Code Atlas Legend</div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-[#f59e0b]"></div>
                                <span className="text-[10px] font-medium opacity-80">Structs</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-[#10b981]"></div>
                                <span className="text-[10px] font-medium opacity-80">Interfaces</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-[#3b82f6]"></div>
                                <span className="text-[10px] font-medium opacity-80">Functions</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-[#6b7280]"></div>
                                <span className="text-[10px] font-medium opacity-80">Variables</span>
                            </div>
                        </div>
                    </div>
                    
                    {!selectedIndex && indexFiles.length > 0 && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="text-center opacity-30">
                                <Icon icon="lucide:network" className="text-6xl mx-auto mb-2" />
                                <p>Select an index to visualize</p>
                            </div>
                        </div>
                    )}

                    {/* Info Drawer (Selected Node or Edge) */}
                    {(selectedNode || selectedEdge) && (
                        <div className="absolute right-4 top-4 bottom-4 w-96 bg-base-200/95 backdrop-blur-xl rounded-2xl border border-base-300 shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-right duration-300 z-20">
                            {selectedNode ? (
                                <>
                                    <div className="p-5 border-b border-base-300 flex justify-between items-start bg-base-300/30">
                                        <div className="min-w-0 pr-4">
                                            <div className="flex items-center gap-2 mb-1">
                                                <div className="badge badge-ghost badge-outline text-[10px] font-mono px-1 h-4 opacity-50 uppercase tracking-tighter">
                                                    {selectedNode.kind}
                                                </div>
                                            </div>
                                            <h2 className="text-xl font-bold leading-none truncate tracking-tight py-1" title={selectedNode.name}>
                                                {selectedNode.name}
                                            </h2>
                                            <div className="flex items-center gap-2 mt-2">
                                                <div className="flex flex-col bg-base-100/50 px-2 py-1 rounded border border-base-300 min-w-[60px]">
                                                    <span className="text-[7px] uppercase opacity-40 font-bold tracking-tighter leading-none mb-1">Score</span>
                                                    <span className="text-xs font-mono font-bold leading-none">{selectedNode.value}</span>
                                                </div>
                                                {selectedNode.members && selectedNode.members.length > 0 && (
                                                    <div className="flex flex-col bg-base-100/50 px-2 py-1 rounded border border-base-300 min-w-[60px]">
                                                        <span className="text-[7px] uppercase opacity-40 font-bold tracking-tighter leading-none mb-1">Members</span>
                                                        <span className="text-xs font-mono font-bold leading-none">{selectedNode.members.length}</span>
                                                    </div>
                                                )}
                                                {selectedNode.loc > 0 && (
                                                    <div className="flex flex-col bg-base-100/50 px-2 py-1 rounded border border-base-300 min-w-[60px]">
                                                        <span className="text-[7px] uppercase opacity-40 font-bold tracking-tighter leading-none mb-1">Lines</span>
                                                        <span className="text-xs font-mono font-bold leading-none">{selectedNode.loc}</span>
                                                    </div>
                                                )}
                                                <div className="text-[10px] font-mono opacity-40 truncate ml-auto" title={selectedNode.path || ''}>
                                                    {selectedNode.path?.split(/[\\/]/).pop() || 'No Path'}
                                                </div>
                                            </div>
                                        </div>
                                        <button onClick={() => setSelectedNode(null)} className="btn btn-ghost btn-xs btn-circle shrink-0">
                                            <Icon icon="lucide:x" />
                                        </button>
                                    </div>
                                    
                                    <div className="flex-1 overflow-y-auto p-5 space-y-6">
                                        {selectedNode.docstring && (
                                            <section>
                                                <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-2">Documentation</h3>
                                                <div className="text-sm opacity-80 bg-base-100/50 p-3 rounded-lg border border-base-300 leading-relaxed font-sans whitespace-pre-wrap selection:bg-primary/30">
                                                    {selectedNode.docstring}
                                                </div>
                                            </section>
                                        )}

                                        {selectedNode.members && selectedNode.members.length > 0 && (
                                            <section>
                                                <div className="flex items-center justify-between mb-2">
                                                    <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30">Members</h3>
                                                    <span className="badge badge-ghost badge-xs opacity-30">{selectedNode.members.length}</span>
                                                </div>
                                                <div className="flex flex-col gap-1 border border-base-300 rounded-lg overflow-hidden bg-base-100/30">
                                                    {selectedNode.members.map((m: any, idx: number) => (
                                                        <button 
                                                            key={idx}
                                                            onClick={() => focusSymbol(m.name)}
                                                            className="flex items-center justify-between p-2.5 hover:bg-primary/10 transition-all text-xs text-left group border-b border-base-300 last:border-0"
                                                        >
                                                            <div className="flex items-center gap-2">
                                                                <Icon icon="lucide:dot" className="opacity-20 group-hover:text-primary group-hover:opacity-100" />
                                                                <span className="opacity-80 group-hover:text-primary font-medium">{m.name}</span>
                                                            </div>
                                                            <span className="opacity-30 text-[9px] uppercase font-mono bg-base-300/50 px-1 rounded">{m.kind}</span>
                                                        </button>
                                                    ))}
                                                </div>
                                            </section>
                                        )}
                                    </div>
                                    
                                    <div className="p-4 bg-base-300/20 border-t border-base-300 flex justify-center">
                                        <button 
                                            onClick={() => focusSymbol(selectedNode.name)}
                                            className="btn btn-primary btn-xs btn-outline w-full gap-2 border-dashed"
                                        >
                                            <Icon icon="lucide:target" />
                                            Recenter Node
                                        </button>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="p-5 border-b border-base-300 flex justify-between items-start bg-base-300/30">
                                        <div>
                                            <div className="badge badge-primary badge-outline text-[10px] font-mono px-1 h-4 opacity-50 uppercase tracking-tighter mb-1">Relationship</div>
                                            <h2 className="text-xl font-bold tracking-tight">Call Logic</h2>
                                        </div>
                                        <button onClick={() => setSelectedEdge(null)} className="btn btn-ghost btn-xs btn-circle shrink-0">
                                            <Icon icon="lucide:x" />
                                        </button>
                                    </div>
                                    <div className="flex-1 p-5 space-y-6 overflow-y-auto">
                                        <section className="bg-base-100/50 rounded-xl border border-base-300 p-4">
                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-3">Source (Caller)</h3>
                                            <div className="font-mono text-xs font-bold text-primary truncate" title={selectedEdge.source}>
                                                {selectedEdge.source.replace('sym:', '')}
                                            </div>
                                        </section>
                                        
                                        <div className="flex justify-center -my-3 relative z-10">
                                            <div className="bg-base-200 p-2 rounded-full border border-base-300 shadow-sm">
                                                <Icon icon="lucide:arrow-down" className="text-primary animate-bounce-slow" />
                                            </div>
                                        </div>

                                        <section className="bg-base-100/50 rounded-xl border border-base-300 p-4">
                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-3">Target (Callee)</h3>
                                            <div className="font-mono text-xs font-bold text-success truncate" title={selectedEdge.target}>
                                                {selectedEdge.target.replace('sym:', '')}
                                            </div>
                                        </section>

                                        <section>
                                            <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-30 mb-2">Dependency Type</h3>
                                            <div className="p-3 bg-primary/5 rounded-lg border border-primary/20 text-sm font-medium italic opacity-80">
                                                {selectedEdge.label}
                                            </div>
                                        </section>
                                    </div>
                                    <div className="p-4 bg-base-300/20 border-t border-base-300 flex flex-col gap-2">
                                        <button 
                                            onClick={() => focusSymbol(selectedEdge.source.replace('sym:', ''))}
                                            className="btn btn-ghost btn-xs w-full gap-2 border border-base-300"
                                        >
                                            <Icon icon="lucide:log-out" className="rotate-180" />
                                            Jump to Caller
                                        </button>
                                        <button 
                                            onClick={() => focusSymbol(selectedEdge.target.replace('sym:', ''))}
                                            className="btn btn-ghost btn-xs w-full gap-2 border border-base-300"
                                        >
                                            <Icon icon="lucide:log-in" />
                                            Jump to Callee
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
