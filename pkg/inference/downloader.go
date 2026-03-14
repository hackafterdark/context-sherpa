package inference

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// WriteCounter counts the number of bytes written and updates progress
type WriteCounter struct {
	Total      int64
	Downloaded int64
	OnProgress func(float64)
}

func (wc *WriteCounter) Write(p []byte) (int, error) {
	n := len(p)
	atomic.AddInt64(&wc.Downloaded, int64(n))
	if wc.Total > 0 {
		wc.OnProgress(float64(wc.Downloaded) / float64(wc.Total))
	}
	return n, nil
}

// Downloader handles background downloading of models
type Downloader struct {
	modelsDir string
	mu        sync.RWMutex
	progress  map[string]float64
}

// NewDownloader creates a new model downloader
func NewDownloader(modelsDir string) *Downloader {
	_ = os.MkdirAll(modelsDir, 0755)
	return &Downloader{
		modelsDir: modelsDir,
		progress:  make(map[string]float64),
	}
}

// DownloadModel downloads a model from a URL
func (d *Downloader) DownloadModel(ctx context.Context, modelID string, url string) error {
	dest := filepath.Join(d.modelsDir, modelID)
	
	// Check if already downloading
	d.mu.RLock()
	_, exists := d.progress[modelID]
	d.mu.RUnlock()
	if exists {
		return fmt.Errorf("model %s is already being downloaded", modelID)
	}

	d.mu.Lock()
	d.progress[modelID] = 0
	d.mu.Unlock()

	defer func() {
		d.mu.Lock()
		delete(d.progress, modelID)
		d.mu.Unlock()
	}()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download: status %d", resp.StatusCode)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	counter := &WriteCounter{
		Total: resp.ContentLength,
		OnProgress: func(p float64) {
			d.mu.Lock()
			d.progress[modelID] = p
			d.mu.Unlock()
		},
	}

	_, err = io.Copy(out, io.TeeReader(resp.Body, counter))
	return err
}

// GetProgress returns the download progress for a model (0-1)
func (d *Downloader) GetProgress(modelID string) (float64, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	p, exists := d.progress[modelID]
	return p, exists
}
