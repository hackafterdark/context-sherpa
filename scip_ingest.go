package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	scip "github.com/sourcegraph/scip/bindings/go/scip"
	wailsRuntime "github.com/wailsapp/wails/v2/pkg/runtime"
	"google.golang.org/protobuf/proto"
)

// IngestIndex parses a SCIP file and stores its symbols/relationships in SQLite.
func (a *App) IngestIndex(scipPath string) error {
	scipPath = a.normalizePath(scipPath)
	// 1. Get/Init local database for this workspace
	localDB, err := a.getLocalDB(scipPath)
	if err != nil {
		return err
	}

	// 2. Check if already indexed and up to date
	info, err := os.Stat(scipPath)
	if err != nil {
		return err
	}
	mtime := info.ModTime().Unix()

	// 2. Check if we already have this index up to date (version 2 includes forward-slash fix)
	var existingMtime int64
	var version int
	err = localDB.QueryRow("SELECT mtime, version FROM scip_indices WHERE scip_path = ?", scipPath).Scan(&existingMtime, &version)
	if err == nil && existingMtime >= mtime && version >= 2 {
		return nil // Up to date
	}

	// 3. Clear existing data for this index to avoid duplicates
	_, _ = localDB.Exec("DELETE FROM scip_symbols WHERE scip_path = ?", scipPath)
	_, _ = localDB.Exec("DELETE FROM scip_relationships WHERE scip_path = ?", scipPath)

	if a.ctx != nil {
		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "processing",
			"message": "Reading 65MB+ SCIP File...",
		})
	}

	data, err := os.ReadFile(scipPath)
	if err != nil {
		return err
	}

	if a.ctx != nil {
		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "processing",
			"message": "Unmarshaling Protobuf...",
		})
	}

	var index scip.Index
	if err := proto.Unmarshal(data, &index); err != nil {
		return fmt.Errorf("failed to parse SCIP index: %w", err)
	}
	data = nil // Free memory

	// 4. Infer language from filename
	language := ""
	filename := filepath.Base(scipPath)
	if strings.HasPrefix(filename, "index-") {
		language = strings.TrimPrefix(filename, "index-")
		language = strings.TrimSuffix(language, ".scip")
	}
	if language == "" {
		language = "go" // Default fallback
	}

	if a.ctx != nil {
		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "processing",
			"message": "Analyzing Architecture...",
		})
	}

	// 3. Architecture Analysis (Pruning & Impact Calculation)
	projectPkg := a.detectProjectPackage(&index)

	// Pruning helpers
	isLocal := func(sym string) bool { return strings.Contains(sym, "local") }
	isAnonymous := func(sym string) bool {
		low := strings.ToLower(sym)
		return strings.Contains(low, "$anon") || strings.Contains(low, "func.") || strings.Contains(low, "lambda$")
	}
	isStdLib := func(sym string) bool {
		if projectPkg == "" { return false }
		low := strings.ToLower(sym)
		stdlibPrefixes := []string{"go/", "typescript/", "python/", "builtin/"}
		if strings.Contains(low, strings.ToLower(projectPkg)) { return false }
		for _, p := range stdlibPrefixes {
			if strings.Contains(low, p) { return true }
		}
		return false
	}

	refCount := make(map[string]int)
	outboundCalls := make(map[string]int)
	symbolToInfo := make(map[string]*scip.SymbolInformation)

	for _, doc := range index.Documents {
		for _, si := range doc.Symbols {
			symbolToInfo[si.Symbol] = si
		}
		var currentScope string
		for _, occ := range doc.Occurrences {
			if isLocal(occ.Symbol) || isAnonymous(occ.Symbol) { continue }
			if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
				currentScope = occ.Symbol
				continue
			}
			refCount[occ.Symbol]++
			if currentScope != "" {
				outboundCalls[currentScope]++
			}
		}
	}
	for _, si := range index.ExternalSymbols {
		symbolToInfo[si.Symbol] = si
	}

	// 4. Persistence
	if a.ctx != nil {
		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "processing",
			"message": "Storing in SQLite...",
		})
	}

	tx, err := localDB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Clean old data for this path
	tx.Exec("DELETE FROM scip_indices WHERE scip_path = ?", scipPath)

	lang := detectLanguage(index.Documents)
	_, err = tx.Exec("INSERT INTO scip_indices (scip_path, mtime, indexed_at, language) VALUES (?, ?, ?, ?)",
		scipPath, mtime, time.Now().Format(time.RFC3339), lang)
	if err != nil { return err }

	var symbolsIngested, relsIngested int

	stmtSym, _ := tx.Prepare(`
		INSERT INTO scip_symbols (
			scip_path, symbol, name, kind, file_path, dir_path, 
			start_line, end_line, parent_symbol, 
			is_local, is_anonymous, is_stdlib, 
			ref_count, outbound_calls, impact_value
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
	defer stmtSym.Close()

	stmtRel, _ := tx.Prepare(`
		INSERT INTO scip_relationships (scip_path, source_symbol, target_symbol) VALUES (?, ?, ?)
	`)
	defer stmtRel.Close()

	// Pre-grouping for parent discovery
	symbolToDescriptors := make(map[string][]*scip.Descriptor)
	byLen := make(map[int][]string)
	for sym := range symbolToInfo {
		parsed, _ := scip.ParseSymbol(sym)
		if len(parsed.Descriptors) > 0 {
			symbolToDescriptors[sym] = parsed.Descriptors
			byLen[len(parsed.Descriptors)] = append(byLen[len(parsed.Descriptors)], sym)
		}
	}

	for _, doc := range index.Documents {
		dir := filepath.ToSlash(filepath.Dir(doc.RelativePath))
		if dir == "." {
			dir = "root"
		}

		for _, occ := range doc.Occurrences {
			if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
				sym := occ.Symbol
				if isLocal(sym) || isAnonymous(sym) { continue }

				parsed, err := scip.ParseSymbol(sym)
				if err != nil || len(parsed.Descriptors) == 0 { continue }

				leaf := parsed.Descriptors[len(parsed.Descriptors)-1]
				if leaf.Suffix == scip.Descriptor_Namespace { continue }

				name := leaf.Name
				kind := "Function"
				switch leaf.Suffix {
				case scip.Descriptor_Type: kind = "Struct"
				case scip.Descriptor_Term: kind = "Variable"
				case scip.Descriptor_Method, scip.Descriptor_Macro: kind = "Function"
				case scip.Descriptor_Parameter, scip.Descriptor_TypeParameter: continue
				}
				if kind == "Struct" && strings.Contains(strings.ToLower(sym), "interface") {
					kind = "Interface"
				}

				// Parent Discovery
				parentSym := ""
				l := len(parsed.Descriptors)
				if l > 1 {
					candidates := byLen[l-1]
					for _, pSym := range candidates {
						pDescriptors := symbolToDescriptors[pSym]
						match := true
						for i := 0; i < l-1; i++ {
							if pDescriptors[i].Name != parsed.Descriptors[i].Name || pDescriptors[i].Suffix != parsed.Descriptors[i].Suffix {
								match = false
								break
							}
						}
						if match {
							parentSym = pSym
							break
						}
					}
				}

				impact := (outboundCalls[sym] * 5) + (refCount[sym] * 10)
				if kind == "Struct" || kind == "Interface" { impact += 20 }
				if impact < 8 { impact = 8 }

				_, err = stmtSym.Exec(
					scipPath, sym, name, kind, doc.RelativePath, dir,
					int(occ.Range[0]+1), int(occ.Range[0]+1), parentSym,
					isLocal(sym), isAnonymous(sym), isStdLib(sym),
					refCount[sym], outboundCalls[sym], impact,
				)
				if err != nil { return err }
				symbolsIngested++
			} else {
				// Relationship logic (for edges)
				// We'll calculate this on the fly or store it. For scale, let's store it.
				// Wait, Pass 4 logic needs currentScope.
				// I'll handle relationships in a separate loop or refine this one.
			}
		}
	}

	// Relationships Pass
	for _, doc := range index.Documents {
		var currentScope string
		for _, occ := range doc.Occurrences {
			if isLocal(occ.Symbol) || isAnonymous(occ.Symbol) { continue }
			if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
				currentScope = occ.Symbol
				continue
			}
			if currentScope != "" && occ.Symbol != "" {
				_, err = stmtRel.Exec(scipPath, currentScope, occ.Symbol)
				if err != nil { return err }
				relsIngested++
			}
		}
	}

	// 5. Update index metadata with version 2
	_, err = tx.Exec("INSERT OR REPLACE INTO scip_indices (scip_path, mtime, indexed_at, language, version) VALUES (?, ?, datetime('now'), ?, 2)", scipPath, mtime, language)
	if err != nil {
		tx.Rollback()
		return err
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	if a.ctx != nil {
		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "complete",
			"message": "Index Ready!",
		})

		wailsRuntime.EventsEmit(a.ctx, "indexing-status", map[string]string{
			"path":    scipPath,
			"status":  "done",
			"message": fmt.Sprintf("Indexed %d symbols and %d relationships", symbolsIngested, relsIngested),
		})
	}

	return nil
}

func (a *App) detectProjectPackage(index *scip.Index) string {
	if len(index.Documents) == 0 { return "" }
	for _, doc := range index.Documents {
		if len(doc.Symbols) > 0 {
			sym := doc.Symbols[0].Symbol
			if idx := strings.Index(sym, " "); idx != -1 {
				parts := strings.Split(sym, " ")
				if len(parts) >= 4 {
					pkg := parts[3]
					if slashIdx := strings.Index(pkg, "/"); slashIdx != -1 {
						return pkg[:slashIdx]
					}
					return pkg
				}
			}
		}
	}
	return ""
}
