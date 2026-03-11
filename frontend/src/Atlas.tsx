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
                                roamDetail: {
                                    zoomLimit: { min: 0.1, max: 10 }
                                },
                                edgeSymbol: ['none', 'arrow'],
                                edgeSymbolSize: [0, 8],
                                edgeLabel: { show: false }, // Remove clutter
                                    label: {
                                        position: 'right',
                                        formatter: '{b}',
                                        show: true,
                                        fontSize: 9,
                                        color: 'rgba(255,255,255,0.4)',
                                        minMargin: 5,
                                        silent: true // Prevents labels from blocking drag/pan
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

    const handleZoomIn = () => {
        if (!chartInstance.current) return;
        const currentZoom = (chartInstance.current.getOption() as any).series[0].zoom || 1;
        chartInstance.current.setOption({
            series: [{ zoom: currentZoom * 1.5 }]
        });
    };

    const handleZoomOut = () => {
        if (!chartInstance.current) return;
        const currentZoom = (chartInstance.current.getOption() as any).series[0].zoom || 1;
        chartInstance.current.setOption({
            series: [{ zoom: currentZoom / 1.5 }]
        });
    };

    const handleReset = () => {
        if (!chartInstance.current) return;
        chartInstance.current.setOption({
            series: [{ 
                center: null, 
                zoom: 1 
            }]
        });
        setSelectedNode(null);
        setSelectedEdge(null);
    };

    const focusSymbol = (query: string) => {
        if (!chartInstance.current) return;
        
        const chart = chartInstance.current;
        const option = chart.getOption() as any;
        const nodes = option.series[0].data;
        
        let idx = nodes.findIndex((n: any) => n.name === query || (n.id && n.id === "sym:" + query));
        if (idx === -1) {
            idx = nodes.findIndex((n: any) => n.name.toLowerCase().includes(query.toLowerCase()));
        }

        if (idx !== -1) {
            const node = nodes[idx];
            
            try {
                const graphData = (chart as any).getModel().getSeriesByIndex(0).getData();
                const layout = graphData.getItemLayout(idx);
                
                if (layout) {
                    chart.setOption({
                        series: [{
                            center: [layout[0], layout[1]],
                            zoom: 2.5
                        }]
                    });

                    chart.dispatchAction({
                        type: 'focusNodeAdjacency',
                        seriesIndex: 0,
                        dataIndex: idx
                    });
                    
                    chart.dispatchAction({
                        type: 'select',
                        seriesIndex: 0,
                        dataIndex: idx
                    });

                    setSelectedNode(node);
                }
            } catch (err) {
                console.error("Failed to get layout coordinates:", err);
            }
        }
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchTerm) {
            focusSymbol(searchTerm);
        }
    };

    return (
        <div className="flex flex-col flex-1 h-full w-full gap-4 animate-in fade-in duration-300 bg-base-100">
            <div className="flex justify-between items-center px-6 pt-4 shrink-0">
                <div className="flex items-center gap-3">
                    <h1 className="text-3xl font-bold font-sans tracking-tight text-base-content/90">Code Atlas</h1>
                    <div className="badge badge-outline badge-sm opacity-40 font-mono border-base-content/20">{workspaceRoot}</div>
                </div>
            </div>

            <div className="flex flex-1 gap-6 px-6 pb-6 overflow-hidden min-h-0">
                {/* Fixed Sidebar: SCIP Files & Search */}
                <div className="w-80 flex flex-col gap-8 shrink-0 border-r border-base-200 pr-6 overflow-y-auto scrollbar-hide">
                    <div className="flex flex-col gap-4">
                        <div className="text-[10px] uppercase tracking-[0.2em] opacity-40 font-black px-1">Symbol Search</div>
                        <form onSubmit={handleSearch} className="relative">
                            <input
                                type="text"
                                placeholder="Search definitions..."
                                className="input input-bordered w-full bg-base-200/40 focus:bg-base-200 transition-all shadow-sm pl-10 h-10 border-base-300 rounded-lg text-sm"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />
                            <Icon icon="lucide:search" className="absolute left-3.5 top-1/2 -translate-y-1/2 opacity-30 w-4 h-4" />
                        </form>
                    </div>
                    
                    <div className="flex flex-col gap-4">
                        <div className="flex items-center justify-between px-1">
                            <div className="text-[10px] uppercase tracking-[0.2em] opacity-50 font-black">Symbolic Indexes</div>
                            <span className="badge badge-ghost badge-sm opacity-40 font-mono scale-90">{indexFiles.length}</span>
                        </div>
                        
                        <div className="flex flex-col gap-2">
                            {indexFiles.length === 0 ? (
                                <div className="p-8 border border-dashed border-base-300 rounded-lg text-center opacity-30 text-xs">
                                    No indexes found
                                </div>
                            ) : (
                                indexFiles.map((file) => (
                                    <button
                                        key={file}
                                        onClick={() => setSelectedIndex(file)}
                                        className={`btn btn-sm justify-start gap-3 border-none shadow-sm h-12 normal-case font-bold ${
                                            selectedIndex === file 
                                                ? 'btn-primary bg-primary text-primary-content ring-1 ring-primary/40' 
                                                : 'bg-base-200/60 hover:bg-base-300 border border-base-300/40'
                                        } transition-all duration-150 rounded-lg group`}
                                    >
                                        <Icon icon="lucide:file-code" className={`w-4 h-4 ${selectedIndex === file ? 'opacity-100' : 'opacity-30 group-hover:opacity-100'} transition-opacity`} />
                                        <div className="flex flex-col items-start min-w-0">
                                            <span className="truncate max-w-[180px] font-mono text-[11px] tracking-tight">{file.split(/[\\/]/).pop()}</span>
                                        </div>
                                    </button>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Main Visualization Area */}
                <div className="flex-1 relative bg-base-100/50 rounded-xl border border-base-200 overflow-hidden shadow-sm flex flex-col min-w-0">
                    {loading && (
                        <div className="absolute inset-0 z-50 flex items-center justify-center bg-base-100/50 backdrop-blur-sm">
                            <span className="loading loading-spinner loading-lg text-primary"></span>
                        </div>
                    )}
                    <div ref={chartRef} className="flex-1 w-full" />
                    
                    {/* Legend & Controls Overlay */}
                    <div className="absolute left-6 bottom-6 flex flex-col gap-4 pointer-events-none z-10">
                        {/* Control Bar */}
                        <div className="flex items-center gap-1 bg-base-100/90 backdrop-blur-xl border border-base-300/50 rounded-xl p-1.5 shadow-xl ring-1 ring-black/5 pointer-events-auto w-fit">
                            <button 
                                onClick={handleZoomIn}
                                className="btn btn-ghost btn-sm btn-square hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60"
                                title="Zoom In"
                            >
                                <Icon icon="lucide:zoom-in" className="w-4 h-4" />
                            </button>
                            <button 
                                onClick={handleZoomOut}
                                className="btn btn-ghost btn-sm btn-square hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60"
                                title="Zoom Out"
                            >
                                <Icon icon="lucide:zoom-out" className="w-4 h-4" />
                            </button>
                            <div className="w-px h-4 bg-base-content/10 mx-0.5" />
                            <button 
                                onClick={handleReset}
                                className="btn btn-ghost btn-sm px-3 gap-2 hover:bg-primary/10 hover:text-primary transition-colors text-base-content/60 text-[10px] font-black uppercase tracking-wider"
                                title="Reset View"
                            >
                                <Icon icon="lucide:refresh-cw" className="w-3.5 h-3.5" />
                                Reset
                            </button>
                        </div>

                        {/* Legend */}
                        <div className="bg-base-100/90 backdrop-blur-xl border border-base-300/50 rounded-xl p-4 shadow-xl ring-1 ring-black/5 pointer-events-auto">
                            <div className="text-[9px] font-black uppercase tracking-[0.2em] opacity-30 mb-3 ml-1">Symbol Legend</div>
                            <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#f59e0b] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">Structs</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#10b981] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">Interfaces</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#3b82f6] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">Functions</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#6b7280] shadow-sm"></div>
                                    <span className="text-[10px] font-bold opacity-70 tracking-tight">Variables</span>
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

                    {/* Info Drawer Overlay */}
                    {(selectedNode || selectedEdge) && (
                        <div className="absolute right-6 top-6 bottom-6 w-[380px] bg-base-200 border border-base-300 shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-right duration-300 z-40 rounded-xl p-0.5">
                            <div className="flex flex-col h-full bg-base-100 rounded-[10px] overflow-hidden">
                            {selectedNode ? (
                                <>
                                    <div className="p-6 border-b border-base-300/50 flex justify-between items-start bg-base-200/50">
                                        <div className="min-w-0 pr-4">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="badge badge-primary badge-outline text-[10px] font-black px-2 h-5 opacity-60 uppercase tracking-widest rounded-md">
                                                    {selectedNode.kind}
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
                                                <div className="text-xs opacity-90 bg-base-200/30 p-4 rounded-xl border border-base-300/50 leading-relaxed font-sans whitespace-pre-wrap italic">
                                                    {selectedNode.docstring}
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
                                                            onClick={() => focusSymbol(m.name)}
                                                            className="flex items-center justify-between p-3.5 hover:bg-primary hover:text-primary-content rounded-lg transition-all duration-200 text-xs text-left group border border-transparent shadow-sm"
                                                        >
                                                            <div className="flex items-center gap-3">
                                                                <Icon icon="lucide:layers" className="opacity-20 group-hover:opacity-100 w-3.5 h-3.5" />
                                                                <span className="font-bold opacity-80 group-hover:opacity-100">{m.name}</span>
                                                            </div>
                                                            <span className="opacity-30 text-[9px] uppercase font-mono bg-base-300/50 px-2 py-0.5 rounded group-hover:bg-primary-focus group-hover:opacity-100">{m.kind}</span>
                                                        </button>
                                                    ))}
                                                </div>
                                            </section>
                                        )}
                                    </div>
                                    
                                    <div className="p-6 bg-base-200/50 border-t border-base-300/50">
                                        <button 
                                            onClick={() => focusSymbol(selectedNode.name)}
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
