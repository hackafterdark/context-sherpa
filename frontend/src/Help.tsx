import { Icon } from '@iconify/react';

const Help = () => {
    const mcpTools = [
        { name: 'list_local_models', desc: 'List available local SLMs and their status.' },
        { name: 'switch_local_model', desc: 'Changes the active model in the Hub.' },
        { name: 'query_local_reasoning', desc: '(The Fallback) Open-ended semantic question tool.' },
        { name: 'classify_repo_intent', desc: '(The Router) Decides which Sherpa tool fits a query best (Symbolic, Structural, or Semantic).' },
        { name: 'summarize_code_intent', desc: '3-sentence functional summary of code (Inputs, Outputs, Side-effects).' },
        { name: 'generate_structural_pattern', desc: 'Translates natural language into ast-grep S-expressions/patterns.' },
        { name: 'analyze_impact_triage', desc: 'Identifies call sites most affected by a change (SCIP-based).' },
        { name: 'check_rule_compliance', desc: 'Validates code against project-specific rules.' },
        { name: 'scan_code', desc: 'Scan code strings for ast-grep rule violations.' },
        { name: 'scan_path', desc: 'Scan files/directories/globs for rule violations.' },
        { name: 'get_symbol_map', desc: 'Definitions and references for a symbol (SCIP).' },
        { name: 'add_or_update_rule', desc: 'Create/update ast-grep rules via YAML.' },
        { name: 'remove_rule', desc: 'Remove an ast-grep rule from the workspace.' },
        { name: 'initialize_ast_grep', desc: 'Setup workspace for ast-grep (sgconfig.yml/rules).' },
        { name: 'search_community_rules', desc: 'Search pre-built rules in the community repo.' },
        { name: 'get_community_rule_details', desc: 'Full YAML content for community rules.' },
        { name: 'import_community_rule', desc: 'Import community rules to local workspace.' },
        { name: 'list_symbols_in_file', desc: 'List classes, functions, and variables in a file.' },
        { name: 'search_definitions', desc: 'Search for symbol definitions project-wide.' },
        { name: 'initialize_scip', desc: 'Index workspace for SCIP-based navigation.' },
    ];

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div>
                <h1 className="text-3xl font-bold mb-2">Help & Documentation</h1>
                <p className="text-base-content/70">Learn how to use Context Sherpa and configure it for your workflow.</p>
            </div>

            <section className="card bg-base-100 shadow-xl overflow-hidden">
                <div className="card-body">
                    <h2 className="card-title text-2xl flex items-center gap-2">
                        <Icon icon="lucide:info" className="text-primary" />
                        Overview
                    </h2>
                    <div className="prose max-w-none text-base-content/80">
                        <p>
                            Context Sherpa is a single executable that functions as both a <strong>GUI application</strong> and an <strong>MCP (Model Context Protocol) Server</strong>.
                        </p>
                        <ul className="list-disc pl-5 space-y-2">
                            <li>
                                <strong>GUI Mode:</strong> The interface you are currently using. It allows you to manage workspaces, browse code, and visualize symbolic relationships.
                            </li>
                            <li>
                                <strong>MCP Server Mode:</strong> A "headless" mode used by AI coding tools (like Cursor, Cline, or Roo Code). It automatically activates when run by an AI agent tool, but can be forced with the <code>--mcp</code> flag.
                            </li>
                        </ul>
                    </div>
                </div>
            </section>

            <section className="card bg-base-100 shadow-xl overflow-hidden">
                <div className="card-body">
                    <h2 className="card-title text-2xl flex items-center gap-2">
                        <Icon icon="lucide:settings" className="text-primary" />
                        MCP Configuration
                    </h2>
                    <p className="text-base-content/80 mb-4">To use Context Sherpa with your AI agent, add it to your tools configuration file (usually <code>mcp_settings.json</code>):</p>
                    <div className="bg-base-300 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                        <pre>{JSON.stringify({
                            mcpServers: {
                                "context-sherpa": {
                                    command: "context-sherpa",
                                    args: ["--projectRoot", "/path/to/your/project"]
                                }
                            }
                        }, null, 2)}</pre>
                    </div>
                </div>
            </section>

            <section className="card bg-base-100 shadow-xl overflow-hidden">
                <div className="card-body">
                    <h2 className="card-title text-2xl flex items-center gap-2">
                        <Icon icon="lucide:layout" className="text-primary" />
                        GUI Screens
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
                        <div className="flex flex-col gap-2 p-4 bg-base-200 rounded-lg">
                            <h3 className="font-bold flex items-center gap-2 text-lg">
                                <Icon icon="lucide:layout-dashboard" className="text-secondary" />
                                Dashboard
                            </h3>
                            <p className="text-sm text-base-content/70">
                                Shows registered workspaces. Workspaces are registered <strong>manually</strong> via the GUI or <strong>automatically</strong> when an AI agent runs the MCP server on a project.
                            </p>
                        </div>
                        <div className="flex flex-col gap-2 p-4 bg-base-200 rounded-lg">
                            <h3 className="font-bold flex items-center gap-2 text-lg">
                                <Icon icon="lucide:network" className="text-secondary" />
                                Code Atlas
                            </h3>
                            <p className="text-sm text-base-content/70">
                                Visualizes your codebase using SCIP. Browse source code, explore symbol relationships, and search for specific functions or classes in a relational graph.
                            </p>
                        </div>
                        <div className="flex flex-col gap-2 p-4 bg-base-200 rounded-lg md:col-span-2">
                            <h3 className="font-bold flex items-center gap-2 text-lg">
                                <Icon icon="lucide:settings" className="text-secondary" />
                                Settings
                            </h3>
                            <p className="text-sm text-base-content/70">
                                Configure your environment and download required dependencies. Use this area to:
                            </p>
                            <ul className="text-sm text-base-content/70 list-disc pl-5 mt-1 space-y-1">
                                <li><strong>SCIP Tools:</strong> Download 3rd party indexers for Go, TypeScript, Python, and more.</li>
                                <li><strong>Local SLMs:</strong> Manage and download small local models for semantic code reasoning.</li>
                                <li><strong>ast-grep:</strong> Install or update the core pattern-matching engine used for code scanning.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            <section className="card bg-base-100 shadow-xl overflow-hidden">
                <div className="card-body p-0">
                    <div className="p-8">
                        <h2 className="card-title text-2xl mb-4 flex items-center gap-2">
                            <Icon icon="lucide:wrench" className="text-primary" />
                            Exposed MCP Tools
                        </h2>
                        <p className="text-base-content/80 font-medium tracking-wide text-sm">The following tools are available to AI agents connected to Context Sherpa.</p>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="table w-full">
                            <thead>
                                <tr className="bg-base-200">
                                    <th className="rounded-none">Tool Name</th>
                                    <th className="rounded-none">Description</th>
                                </tr>
                            </thead>
                            <tbody>
                                {mcpTools.map((tool) => (
                                    <tr key={tool.name} className="hover:bg-base-200/50 transition-colors">
                                        <td className="font-mono text-secondary text-sm font-semibold">{tool.name}</td>
                                        <td className="text-sm text-base-content/80">{tool.desc}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Help;
