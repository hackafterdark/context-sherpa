package inference

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloader_DownloadModel(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "models-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("mock-wasm-content"))
	}))
	defer ts.Close()

	d := NewDownloader(tempDir)
	err = d.DownloadModel(context.Background(), "test-model", ts.URL)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify file exists
	path := filepath.Join(tempDir, "test-model")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("expected model file to exist at %s", path)
	}

	content, _ := os.ReadFile(path)
	if string(content) != "mock-wasm-content" {
		t.Errorf("expected content 'mock-wasm-content', got '%s'", string(content))
	}
}
