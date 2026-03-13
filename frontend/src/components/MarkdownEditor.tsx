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
    linkPlugin
} from '@mdxeditor/editor';
import '@mdxeditor/editor/style.css';
import { useRef, useEffect } from 'react';

type MarkdownEditorProps = {
    markdown: string;
    onChange: (markdown: string) => void;
    readOnly?: boolean;
};

// Custom styles to ensure headings look distinct and the editor feels premium
const editorStyles = `
  .mdxeditor-root {
    --accent-color: var(--color-primary);
  }
  .mdxeditor-toolbar {
    background-color: var(--color-base-300) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    position: sticky;
    top: 0;
    z-index: 20;
    padding: 0.5rem 1rem !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .mdxeditor-toolbar button { color: var(--color-base-content) !important; opacity: 0.7; }
  .mdxeditor-toolbar button:hover { opacity: 1; background-color: rgba(255, 255, 255, 0.05) !important; }
  .mdxeditor-toolbar button[data-active="true"] { color: var(--color-primary) !important; background-color: rgba(var(--color-primary-rgb), 0.1) !important; opacity: 1; }
  
  .mdxeditor-toolbar select, .mdxeditor-toolbar [role="combobox"] { 
    background-color: var(--color-base-100) !important; 
    color: var(--color-base-content) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    min-width: 120px;
  }

  /* Popups (Radix) - aggressive overrides to fix contrast */
  [data-radix-popper-content-wrapper] { 
    z-index: 9999 !important; 
  }
  
  /* Target the content container inside the popper */
  [data-radix-popper-content-wrapper] [role="menu"],
  [data-radix-popper-content-wrapper] [role="listbox"],
  .mdxeditor-popup-container { 
    background-color: #1a1b1e !important; /* Force a dark background if base-300 is failing */
    background: #1a1b1e !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7) !important;
    padding: 4px !important;
    border-radius: 8px !important;
    color: #e5e7eb !important; /* Force light text */
  }
  
  /* Target items inside the popup */
  [data-radix-popper-content-wrapper] [role="option"],
  [data-radix-popper-content-wrapper] [role="menuitem"],
  [data-radix-popper-content-wrapper] [data-radix-collection-item],
  .mdxeditor-popup-container [role="option"],
  .mdxeditor-popup-container [role="menuitem"] {
    color: #e5e7eb !important; /* Light text */
    background-color: transparent !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    font-size: 0.875rem !important;
    outline: none !important;
  }

  /* Hover/Select states */
  [data-radix-popper-content-wrapper] [role="option"]:hover,
  [data-radix-popper-content-wrapper] [role="menuitem"]:hover,
  [data-radix-popper-content-wrapper] [data-highlighted],
  [data-radix-popper-content-wrapper] [data-state="selected"],
  .mdxeditor-popup-container [role="option"]:hover,
  .mdxeditor-popup-container [data-highlighted] {
    background-color: var(--color-primary) !important;
    color: var(--color-primary-content) !important;
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
    font-size: 0.9em;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  }
`;

export default function MarkdownEditor({ markdown, onChange, readOnly = false }: MarkdownEditorProps) {
    const editorRef = useRef<MDXEditorMethods>(null);

    // Sync markdown prop with editor if it changes externally
    useEffect(() => {
        if (editorRef.current) {
            editorRef.current.setMarkdown(markdown);
        }
    }, [markdown]);

    return (
        <div className="h-full w-full flex flex-col overflow-hidden bg-base-100 rounded-lg border border-base-content/10 shadow-xl overflow-y-auto scrollbar-thin scrollbar-thumb-base-content/10">
            <style>{editorStyles}</style>
            <MDXEditor
                ref={editorRef}
                markdown={markdown}
                onChange={onChange}
                readOnly={readOnly}
                plugins={[
                    headingsPlugin(),
                    listsPlugin(),
                    quotePlugin(),
                    thematicBreakPlugin(),
                    markdownShortcutPlugin(),
                    codeBlockPlugin(),
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
                                <CreateLink />
                                <InsertCodeBlock />
                            </>
                        )
                    })
                ]}
                className="dark-theme flex-1"
                contentEditableClassName="prose prose-invert max-w-none outline-none p-8 text-base-content"
            />
        </div>
    );
}


