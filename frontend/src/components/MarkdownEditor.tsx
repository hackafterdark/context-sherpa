import {
  MDXEditor,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  codeBlockPlugin,
  toolbarPlugin,
  BlockTypeSelect,
  BoldItalicUnderlineToggles,
  ListsToggle,
  UndoRedo,
  CreateLink,
  InsertCodeBlock,
  CodeToggle,
  MDXEditorMethods,
  linkPlugin,
  codeMirrorPlugin,
  CodeMirrorEditor,
  frontmatterPlugin,
  InsertFrontmatter,
  tablePlugin,
  imagePlugin,
  InsertTable,
  InsertImage
} from '@mdxeditor/editor';
import '@mdxeditor/editor/style.css';
import { useRef, Component, ErrorInfo, ReactNode, useEffect } from 'react';
import { Icon } from '@iconify/react';

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(error: Error, errorInfo: ErrorInfo) { console.error("MDXEditor Crash:", error, errorInfo); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center bg-error/10 text-error p-8 rounded-xl border border-error/20 h-full min-h-[300px]">
          <Icon icon="lucide:alert-circle" className="text-4xl mb-2" />
          <h2 className="font-bold">Editor Rendering Failed</h2>
          <p className="text-xs opacity-70 mt-2 text-center max-w-xs">This file may contain complex markdown that the editor cannot process. Try a simpler rule format.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

type MarkdownEditorProps = {
  markdown: string;
  onChange: (markdown: string) => void;
  theme: string;
  readOnly?: boolean;
};

// Custom styles to ensure headings look distinct and the editor feels premium
const editorStyles = `
  .mdxeditor {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden;
    overflow-y: auto !important;
    min-height: 0 !important;
    background-color: transparent !important;
  }
  .mdxeditor-root {
    --accent-color: var(--color-primary);
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    overflow: hidden !important;
    background-color: transparent !important;
  }
  .mdxeditor-rich-text {
    flex: 1 !important;
    overflow-y: auto !important;
    padding-bottom: 20vh !important;
    outline: none !important;
    position: relative !important;
    scrollbar-width: thin;
    scrollbar-color: rgba(var(--color-primary-rgb), 0.3) transparent;
  }
  
  /* Critical for ensuring the editor occupies the full flex space and allows scrolling */
  .mdxeditor-rich-text > div:first-child {
    min-height: 100% !important;
    display: block !important;
    overflow: visible !important;
  }

  /* The actual content area inside the rich-text div */
  .mdxeditor-rich-text [contenteditable="true"] {
    min-height: 100% !important;
    padding-bottom: 25rem !important; /* Extra padding to ensure scroll always works */
    cursor: text !important;
  }

  .mdxeditor-toolbar {
    background-color: var(--color-base-300) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 0.5rem 1rem !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    scrollbar-width: none;
    z-index: 20;
    flex-shrink: 0 !important;
  }
  .mdxeditor-toolbar::-webkit-scrollbar { display: none; }

  .mdxeditor-toolbar button { color: var(--color-base-content) !important; opacity: 0.7; }
  .mdxeditor-toolbar button:hover { opacity: 1; background-color: rgba(255, 255, 255, 0.05) !important; }
  .mdxeditor-toolbar button[data-active="true"] { color: var(--color-primary) !important; background-color: rgba(var(--color-primary-rgb), 0.1) !important; opacity: 1; }
  
  /* Select boxes and code-block language selector */
  .mdxeditor-toolbar select, 
  .mdxeditor-toolbar [role="combobox"],
  [data-mdx-editor-code-block-language-select] select,
  .mdxeditor select,
  .mdxeditor-code-block-language-select select { 
    background-color: var(--color-base-200) !important;
    color: var(--color-base-content) !important;
    // border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    padding: 0.25rem 0.6rem !important;
    min-width: 100px;
    cursor: pointer;
  }
    
  .mdxeditor [role="combobox"] {
    color: var(--color-base-content) !important;
    background-color: var(--color-base-200) !important;
  }

  .mdxeditor-select-content {
    color: var(--color-base-content) !important;
    background-color: var(--color-base-200) !important;
  } 

  /* Specific fix for the language dropdown in code blocks - matching their structure */
  [class*="CodeBlockEditor_languageSelect"] {
    background-color: #1a1b1e !important;
    color: #f3f4f6 !important;
    border: 1px solid #444 !important;
  }

  /* MDXEditor Theming - Mapping DaisyUI 5 to Radix variables */
  .mdxeditor-theme-wrapper {
    --baseBase: var(--color-base-100) !important;
    --baseBgSubtle: var(--color-base-200) !important;
    --baseBg: var(--color-base-100) !important;
    --baseBgHover: var(--color-base-300) !important;
    --baseBgActive: var(--color-base-100) !important;
    --baseLine: var(--color-base-300) !important;
    --baseBorder: var(--color-base-300) !important;
    --baseBorderHover: var(--color-base-content) !important;
    --baseSolid: var(--color-base-300) !important;
    --baseSolidHover: var(--color-base-200) !important;
    --baseText: var(--color-base-content) !important;
    --baseTextContrast: var(--color-base-100) !important;
    
    --accentBase: var(--color-primary) !important;
    --accentBgSubtle: color-mix(in srgb, var(--color-primary) 15%, transparent) !important;
    --accentBg: color-mix(in srgb, var(--color-primary) 30%, transparent) !important;
    --accentBgHover: color-mix(in srgb, var(--color-primary) 45%, transparent) !important;
    --accentBgActive: color-mix(in srgb, var(--color-primary) 60%, transparent) !important;
    --accentLine: var(--color-primary) !important;
    --accentBorder: var(--color-primary) !important;
    --accentBorderHover: var(--color-primary) !important;
    --accentSolid: var(--color-primary) !important;
    --accentSolidHover: var(--color-primary) !important;
    --accentText: var(--color-base-content) !important;
    --accentTextContrast: var(--color-primary-content) !important;
  }

  /* Portal Styling - Ensuring legality across all themes */
  [data-radix-portal] [role="listbox"],
  [data-radix-portal] [role="menu"],
  [data-radix-portal] [role="combobox"],
  [data-radix-portal] [data-radix-select-content],
  [data-radix-portal] [data-radix-popper-content-wrapper],
  .mdxeditor-popup-container { 
    background-color: var(--color-base-200) !important;
    border: 1px solid var(--color-base-300) !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5) !important;
    padding: 6px !important;
    border-radius: 8px !important;
    color: var(--color-base-content) !important;
    z-index: 99999 !important;
  }

  /* Prevent invisible portal containers from adding height */
  .mdxeditor-popup-container:empty {
    display: none !important;
  }

  /* Ensure the popup container itself stays out of document flow */
  .mdxeditor-popup-container {
    position: absolute !important;
    top: 0;
    left: 0;
  }

  [data-radix-portal] [role="option"],
  [data-radix-portal] [role="menuitem"],
  [data-radix-portal] * {
    color: var(--color-base-content) !important;
    font-size: 0.85rem !important;
  }

  [data-radix-portal] [role="option"][data-highlighted],
  [data-radix-portal] [role="menuitem"][data-highlighted],
  [data-radix-portal] [data-state="checked"] {
    background-color:  var(--color-primary) !important;
    color: var(--color-primary-content) !important;
  }

  /* CodeMirror background should match editor bg in light mode, but stay dark in dark modes */
  .dark-theme .cm-editor {
    background-color: #1a1b1e !important;
  }
  
  /* Fallback for light mode code editor */
  .mdxeditor-theme-wrapper:not(.dark-theme) .cm-editor {
    background-color: var(--color-base-200) !important;
    color: var(--color-base-content) !important;
  }

  .mdxeditor-theme-wrapper .dark-theme .cm-editor {
    background-color: var(--color-base-200) !important;
    color: color(display-p3 0.93 0.933 0.94)
  }

  /* Extreme fix for native selects inside the editor */
  select option {
    background-color: var(--color-base-200) !important;
    color: var(--color-base-content) !important;
  }

  .prose h1 { font-size: 2.25rem; font-weight: 800; margin-top: 1.5rem; color: var(--color-primary); }
  .prose h2 { font-size: 1.875rem; font-weight: 700; margin-top: 1.25rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 0.5rem; }
  .prose h3 { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; }
  .prose h4 { font-size: 1.25rem; font-weight: 600; }
  .prose p { margin-top: 0.75rem; line-height: 1.75; }
  .prose ul { margin-top: 0.75rem; list-style-type: disc; padding-left: 1.5rem; }
  .prose li { margin-top: 0.25rem; }
  
  /* Inline code highlighting */
  .prose :not(pre) > code {
    background-color: var(--color-base-300);
    color: var(--color-primary);
    padding: 0.125rem 0.25rem;
    border-radius: 0.375rem;
    font-size: 1em;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  }

  /* Code block styling */
  .cm-editor {
    background-color: #1a1b1e !important;
    border-radius: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }
`;

export default function MarkdownEditor({ markdown, onChange, theme, readOnly = false }: MarkdownEditorProps) {
  const editorRef = useRef<MDXEditorMethods>(null);

  // Robust normalization for equality check
  const normalizeForComparison = (text: string) => {
    if (!text) return '';
    return text
      .replace(/\r\n|\r/g, '\n') // Line endings
      .replace(/\s+$/gm, '')     // Trailing spaces per line
      .trim();                  // Flanking whitespace
  };

  useEffect(() => {
    console.log("MDXEditor: Prop 'markdown' length:", markdown?.length || 0);
    if (editorRef.current) {
      const currentMarkdown = editorRef.current.getMarkdown();
      console.log("MDXEditor: Internal state length:", currentMarkdown.length);

      // Only force sync if the content is meaningfully different
      if (normalizeForComparison(currentMarkdown) !== normalizeForComparison(markdown)) {
        console.log("MDXEditor: Force syncing content (meaningful difference detected)");
        editorRef.current.setMarkdown(markdown);
      } else {
        // Even if slightly different (whitespace), don't force sync if it's "equivalent"
        // to avoid infinite loops or focus loss
        // console.log("MDXEditor: Content equivalent, skipping force sync");
      }
    }
  }, [markdown]);

  return (
    <ErrorBoundary>
      <div className={`flex-1 w-full flex flex-col bg-base-100 min-h-0 overflow-hidden relative border border-base-content/10 rounded-xl mdxeditor-theme-wrapper ${theme === 'light' ? '' : 'dark-theme'}`}>
        <style>{editorStyles}</style>
        <MDXEditor
          ref={editorRef}
          markdown={markdown}
          onChange={onChange}
          readOnly={readOnly}
          suppressHtmlProcessing={true}
          plugins={[
            headingsPlugin(),
            listsPlugin(),
            quotePlugin(),
            thematicBreakPlugin(),
            markdownShortcutPlugin(),
            tablePlugin(),
            frontmatterPlugin(),
            imagePlugin(),
            codeBlockPlugin({
              codeBlockEditorDescriptors: [
                {
                  priority: -10,
                  match: () => true,
                  Editor: CodeMirrorEditor
                }
              ]
            }),
            codeMirrorPlugin({ codeBlockLanguages: { js: 'JavaScript', ts: 'TypeScript', tsx: 'TypeScript (react)', jsx: 'JavaScript (react)', css: 'CSS', go: 'Go', python: 'Python', bash: 'Bash' } }),
            linkPlugin(),
            toolbarPlugin({
              toolbarContents: () => (
                <>
                  <UndoRedo />
                  <div className="w-px h-6 bg-base-content/10 mx-1" />
                  <BlockTypeSelect />
                  <BoldItalicUnderlineToggles />
                  <CodeToggle />
                  <div className="w-px h-6 bg-base-content/10 mx-1" />
                  <ListsToggle />
                  <div className="w-px h-6 bg-base-content/10 mx-1" />
                  <InsertFrontmatter />
                  <div className="w-px h-6 bg-base-content/10 mx-1" />
                  <CreateLink />
                  <InsertImage />
                  <InsertTable />
                  <InsertCodeBlock />
                </>
              )
            })
          ]}
          className={`flex-1 min-h-0 ${theme === 'light' ? '' : 'dark-theme'}`}
          contentEditableClassName={`prose max-w-none outline-none p-12 pb-[50vh] text-base-content scroll-mt-20 ${theme === 'light' ? '' : 'prose-invert'}`}
        />
      </div>
    </ErrorBoundary>
  );
}


